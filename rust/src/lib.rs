//! A Rust reimplementation of mini-SWE-agent with an emphasis on a small,
//! explicit architecture and low per-step overhead.
//!
//! The design mirrors the Python reference:
//! - [`Model`] is the language-model boundary.
//! - [`Environment`] is the command-execution boundary.
//! - [`DefaultAgent`] is the linear control loop.
//!
//! The crate deliberately exposes concrete traits and plain data structures
//! instead of a framework.  That keeps custom models and environments easy to
//! implement while still allowing the hot path to avoid unnecessary clones.

pub mod agent;
pub mod config;
pub mod environments;
pub mod models;
pub mod run;
pub mod template;

pub use agent::{Agent, AgentConfig, AgentMode, DefaultAgent, InteractiveAgent};
pub use config::{get_config_from_spec, recursive_merge};
pub use environments::{DockerEnvironment, LocalEnvironment};
pub use models::{LlmConnectorModel, ModelConfig, TextBasedModel};

use std::sync::{Mutex, OnceLock};

use anyhow::Result as AnyResult;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// A chat message with mini-SWE-agent's `extra` metadata attached.
///
/// `fields` keeps the representation open: providers may add tool calls,
/// response-API items, or other protocol-specific keys, and they round-trip
/// through serialization without maintaining a second message type.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Message {
    pub role: String,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub content: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<Map<String, Value>>,
    #[serde(flatten)]
    pub fields: Map<String, Value>,
}

impl Message {
    pub fn new(role: impl Into<String>, content: impl Into<Value>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            extra: None,
            fields: Map::new(),
        }
    }

    pub fn system(content: impl Into<Value>) -> Self {
        Self::new("system", content)
    }

    pub fn user(content: impl Into<Value>) -> Self {
        Self::new("user", content)
    }

    pub fn assistant(content: impl Into<Value>) -> Self {
        Self::new("assistant", content)
    }

    pub fn with_extra(mut self, extra: Map<String, Value>) -> Self {
        self.extra = Some(extra);
        self
    }

    pub fn ensure_extra(&mut self) -> &mut Map<String, Value> {
        self.extra.get_or_insert_with(Map::new)
    }

    pub fn extra_value(&self, key: &str) -> Option<&Value> {
        self.extra.as_ref().and_then(|extra| extra.get(key))
    }

    pub fn set_extra(&mut self, key: impl Into<String>, value: Value) {
        self.ensure_extra().insert(key.into(), value);
    }

    pub fn actions(&self) -> std::result::Result<Vec<Action>, serde_json::Error> {
        let Some(value) = self.extra_value("actions") else {
            return Ok(Vec::new());
        };
        serde_json::from_value(value.clone())
    }

    pub fn exit_status(&self) -> &str {
        self.extra_value("exit_status")
            .and_then(Value::as_str)
            .unwrap_or("")
    }

    pub fn submission(&self) -> &str {
        self.extra_value("submission")
            .and_then(Value::as_str)
            .unwrap_or("")
    }

    pub fn into_api_value(self) -> std::result::Result<Value, serde_json::Error> {
        let mut value = serde_json::to_value(self)?;
        if let Some(object) = value.as_object_mut() {
            object.remove("extra");
        }
        Ok(value)
    }

    pub fn api_value(&self) -> std::result::Result<Value, serde_json::Error> {
        self.clone().into_api_value()
    }

    pub fn exit_message(
        kind: &str,
        content: impl Into<Value>,
        submission: impl Into<String>,
    ) -> Self {
        let mut message = Self::new("exit", content);
        message.set_extra("exit_status", Value::String(kind.to_string()));
        message.set_extra("submission", Value::String(submission.into()));
        message
    }
}

/// A shell action produced by a model.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Action {
    pub command: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Action {
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            tool_call_id: None,
        }
    }

    pub fn with_tool_call_id(command: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

/// Result of executing one action.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Output {
    pub output: String,
    pub returncode: i32,
    #[serde(default)]
    pub exception_info: String,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub extra: Map<String, Value>,
}

impl Output {
    pub fn success(output: impl Into<String>, returncode: i32) -> Self {
        Self {
            output: output.into(),
            returncode,
            exception_info: String::new(),
            extra: Map::new(),
        }
    }

    pub fn failure(
        error: impl std::fmt::Display,
        exception_type: &str,
        output: impl Into<String>,
    ) -> Self {
        let mut extra = Map::new();
        extra.insert(
            "exception_type".to_string(),
            Value::String(exception_type.to_string()),
        );
        extra.insert("exception".to_string(), Value::String(error.to_string()));
        Self {
            output: output.into(),
            returncode: -1,
            exception_info: format!("An error occurred while executing the command: {error}"),
            extra,
        }
    }
}

/// Interrupt reasons used to end or redirect the agent loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterruptKind {
    Submitted,
    LimitsExceeded,
    TimeExceeded,
    UserInterruption,
    FormatError,
    UserNewTask,
    UserRejection,
    Other,
}

impl InterruptKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Submitted => "Submitted",
            Self::LimitsExceeded => "LimitsExceeded",
            Self::TimeExceeded => "TimeExceeded",
            Self::UserInterruption => "UserInterruption",
            Self::FormatError => "FormatError",
            Self::UserNewTask => "UserNewTask",
            Self::UserRejection => "UserRejection",
            Self::Other => "Interrupt",
        }
    }
}

impl std::fmt::Display for InterruptKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A controlled interruption carrying messages that should be added to history.
#[derive(Debug, Error)]
#[error("agent flow interrupted: {kind}")]
pub struct FlowInterrupt {
    pub kind: InterruptKind,
    pub messages: Vec<Message>,
}

impl FlowInterrupt {
    pub fn new(kind: InterruptKind, messages: Vec<Message>) -> Self {
        Self { kind, messages }
    }

    pub fn submitted(submission: impl Into<String>) -> Self {
        let submission = submission.into();
        Self::new(
            InterruptKind::Submitted,
            vec![Message::exit_message(
                "Submitted",
                submission.clone(),
                submission,
            )],
        )
    }

    pub fn limits_exceeded() -> Self {
        Self::new(
            InterruptKind::LimitsExceeded,
            vec![Message::exit_message(
                "LimitsExceeded",
                "LimitsExceeded",
                "",
            )],
        )
    }

    pub fn time_exceeded() -> Self {
        Self::new(
            InterruptKind::TimeExceeded,
            vec![Message::exit_message("TimeExceeded", "TimeExceeded", "")],
        )
    }
}

/// A model output that did not satisfy the expected action format.
#[derive(Debug, Error)]
#[error("model response format error")]
pub struct FormatError {
    pub messages: Vec<Message>,
}

impl FormatError {
    pub fn new(messages: Vec<Message>) -> Self {
        Self { messages }
    }
}

/// Unified error type for the agent, model, and environment boundaries.
#[derive(Debug, Error)]
pub enum AgentError {
    #[error(transparent)]
    Format(#[from] FormatError),
    #[error(transparent)]
    Interrupt(#[from] FlowInterrupt),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl AgentError {
    pub fn other(error: impl Into<anyhow::Error>) -> Self {
        Self::Other(error.into())
    }
}

pub type Result<T> = std::result::Result<T, AgentError>;
pub type AnyhowResult<T> = AnyResult<T>;

/// Model boundary.  Implementations must persist cost information on
/// [`FormatError`] because the billed call happened before parsing failed.
#[async_trait]
pub trait Model: Send + Sync {
    fn model_name(&self) -> &str;

    async fn query(&self, messages: &[Message], kwargs: Option<Value>) -> Result<Message>;

    fn format_message(&self, message: Message) -> Result<Message> {
        Ok(message)
    }

    fn format_observation_messages(
        &self,
        message: &Message,
        outputs: &[Output],
        template_vars: &Value,
    ) -> Result<Vec<Message>>;

    fn get_template_vars(&self) -> Value;

    fn serialize(&self) -> Value;
}

/// Environment boundary.
#[async_trait]
pub trait Environment: Send + Sync {
    async fn execute(
        &self,
        action: &Action,
        cwd: Option<&str>,
        timeout: Option<u64>,
    ) -> Result<Output>;

    fn get_template_vars(&self) -> Value;

    fn serialize(&self) -> Value;

    fn cleanup(&self) -> AnyResult<()> {
        Ok(())
    }
}

/// Global process-wide model stats, matching `MSWEA_GLOBAL_*` limits.
#[derive(Debug)]
pub struct GlobalModelStats {
    pub cost: f64,
    pub n_calls: u64,
    pub cost_limit: f64,
    pub call_limit: u64,
}

impl GlobalModelStats {
    pub fn from_env() -> Self {
        let cost_limit = std::env::var("MSWEA_GLOBAL_COST_LIMIT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0.0);
        let call_limit = std::env::var("MSWEA_GLOBAL_CALL_LIMIT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0);
        Self {
            cost: 0.0,
            n_calls: 0,
            cost_limit,
            call_limit,
        }
    }

    pub fn add(&mut self, cost: f64) -> AnyResult<()> {
        self.cost += cost;
        self.n_calls += 1;
        if (self.cost_limit > 0.0 && self.cost > self.cost_limit)
            || (self.call_limit > 0 && self.n_calls > self.call_limit)
        {
            anyhow::bail!(
                "Global cost/call limit exceeded: ${:.4} / {} calls",
                self.cost,
                self.n_calls
            );
        }
        Ok(())
    }
}

static GLOBAL_MODEL_STATS: OnceLock<Mutex<GlobalModelStats>> = OnceLock::new();

pub fn global_model_stats() -> &'static Mutex<GlobalModelStats> {
    GLOBAL_MODEL_STATS.get_or_init(|| Mutex::new(GlobalModelStats::from_env()))
}

/// Stored under `info.global_model_stats` in serialized trajectories.
pub fn global_stats_snapshot() -> Value {
    let stats = global_model_stats()
        .lock()
        .expect("global model stats lock poisoned");
    serde_json::json!({
        "cost": stats.cost,
        "n_calls": stats.n_calls,
        "cost_limit": stats.cost_limit,
        "call_limit": stats.call_limit,
    })
}

pub(crate) fn json_merge(base: Value, patch: Value) -> Value {
    match (base, patch) {
        (Value::Object(mut base), Value::Object(patch)) => {
            for (key, value) in patch {
                if value.is_null() {
                    continue;
                }
                if let Some(existing) = base.remove(&key) {
                    base.insert(key, json_merge(existing, value));
                } else {
                    base.insert(key, value);
                }
            }
            Value::Object(base)
        }
        (_, patch) => patch,
    }
}
