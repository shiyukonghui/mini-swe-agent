//! The linear agent control loop.
//!
//! The loop is intentionally tiny: query the model, execute every returned
//! action, append observations, and repeat until an exit message appears.
//! Interrupts are ordinary values so the hot path stays explicit and easy to
//! instrument.

use std::path::{Path, PathBuf};
use std::time::Instant;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{
    global_stats_snapshot, json_merge, Action, AgentError, Environment, FlowInterrupt,
    InterruptKind, Message, Model, Result, VERSION,
};

/// Configuration shared by the default and interactive agents.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AgentConfig {
    pub system_template: String,
    pub instance_template: String,
    pub step_limit: u64,
    pub cost_limit: f64,
    pub wall_time_limit_seconds: u64,
    pub max_consecutive_format_errors: u64,
    pub output_path: Option<PathBuf>,
    pub mode: AgentMode,
    pub whitelist_actions: Vec<String>,
    pub confirm_exit: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            system_template: String::new(),
            instance_template: String::new(),
            step_limit: 0,
            cost_limit: 3.0,
            wall_time_limit_seconds: 0,
            max_consecutive_format_errors: 3,
            output_path: None,
            mode: AgentMode::Confirm,
            whitelist_actions: Vec::new(),
            confirm_exit: true,
        }
    }
}

/// Interaction mode for the confirming agent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AgentMode {
    Human,
    #[default]
    Confirm,
    Yolo,
}

/// Agent interface kept object-safe so run scripts can select an
/// implementation dynamically without branching through the control flow.
#[async_trait]
pub trait Agent: Send + Sync {
    async fn run(&mut self, task: &str, kwargs: Option<Value>) -> Result<Value>;
    fn save(&self, path: Option<&Path>, extra_dicts: &[Value]) -> Result<Value>;
}

/// The minimal benchmark agent.
pub struct DefaultAgent {
    pub config: AgentConfig,
    pub messages: Vec<Message>,
    pub model: Box<dyn Model>,
    pub env: Box<dyn Environment>,
    pub extra_template_vars: Map<String, Value>,
    pub cost: f64,
    pub n_calls: u64,
    pub n_consecutive_format_errors: u64,
    start_time: Instant,
}

impl DefaultAgent {
    pub fn new(model: Box<dyn Model>, env: Box<dyn Environment>, config: AgentConfig) -> Self {
        Self {
            config,
            messages: Vec::new(),
            model,
            env,
            extra_template_vars: Map::new(),
            cost: 0.0,
            n_calls: 0,
            n_consecutive_format_errors: 0,
            start_time: Instant::now(),
        }
    }

    pub fn elapsed_seconds(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    pub fn get_template_vars(&self, kwargs: Option<&Value>) -> Value {
        let mut values = vec![
            serde_json::to_value(&self.config).unwrap_or(Value::Null),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            serde_json::json!({
                "n_model_calls": self.n_calls,
                "model_cost": self.cost,
                "elapsed_seconds": self.elapsed_seconds(),
            }),
            Value::Object(self.extra_template_vars.clone()),
        ];
        if let Some(kwargs) = kwargs {
            values.push(kwargs.clone());
        }
        values
            .into_iter()
            .fold(Value::Object(Map::new()), |base, value| {
                json_merge(base, value)
            })
    }

    pub fn render_template(&self, template: &str, kwargs: Option<&Value>) -> Result<String> {
        crate::template::render(template, &self.get_template_vars(kwargs))
    }

    pub fn add_messages(&mut self, messages: Vec<Message>) -> Vec<Message> {
        self.messages.extend(messages.iter().cloned());
        messages
    }

    pub fn handle_uncaught_exception(&mut self, error: &anyhow::Error) -> Vec<Message> {
        let mut extra = Map::new();
        extra.insert(
            "exit_status".to_string(),
            Value::String("RuntimeError".to_string()),
        );
        extra.insert("submission".to_string(), Value::String(String::new()));
        extra.insert(
            "exception_str".to_string(),
            Value::String(error.to_string()),
        );
        extra.insert(
            "traceback".to_string(),
            Value::String(std::backtrace::Backtrace::force_capture().to_string()),
        );
        let message = Message::new("exit", error.to_string()).with_extra(extra);
        self.add_messages(vec![self
            .model
            .format_message(message)
            .unwrap_or_else(|_| Message::new("exit", error.to_string()))])
    }

    pub async fn step(&mut self) -> Result<Vec<Message>> {
        let message = self.query().await?;
        self.execute_actions(&message).await
    }

    pub fn check_limits(&self) -> Result<()> {
        if (self.config.step_limit > 0 && self.n_calls >= self.config.step_limit)
            || (self.config.cost_limit > 0.0 && self.cost >= self.config.cost_limit)
        {
            return Err(FlowInterrupt::limits_exceeded().into());
        }
        if self.config.wall_time_limit_seconds > 0
            && self.elapsed_seconds() >= self.config.wall_time_limit_seconds
        {
            return Err(FlowInterrupt::time_exceeded().into());
        }
        Ok(())
    }

    pub async fn query(&mut self) -> Result<Message> {
        self.check_limits()?;
        self.n_calls += 1;
        let message = self.model.query(&self.messages, None).await?;
        self.cost += message
            .extra_value("cost")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        self.messages.push(message.clone());
        Ok(message)
    }

    pub async fn execute_actions(&mut self, message: &Message) -> Result<Vec<Message>> {
        let actions = message.actions().map_err(AgentError::other)?;
        let mut outputs = Vec::with_capacity(actions.len());
        for action in &actions {
            outputs.push(self.env.execute(action, None, None).await?);
        }
        let template_vars = self.get_template_vars(None);
        let observations =
            self.model
                .format_observation_messages(message, &outputs, &template_vars)?;
        Ok(self.add_messages(observations))
    }

    pub fn serialize(&self, extra_dicts: &[Value]) -> Value {
        let last_message = self.messages.last();
        let exit_status = last_message.map(Message::exit_status).unwrap_or("");
        let submission = last_message.map(Message::submission).unwrap_or("");
        let agent_data = serde_json::json!({
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                },
                "config": {
                    "agent": serde_json::to_value(&self.config).unwrap_or(Value::Null),
                    "agent_type": "mini_swe_agent.agent.DefaultAgent",
                },
                "mini_version": VERSION,
                "exit_status": exit_status,
                "submission": submission,
                "global_model_stats": global_stats_snapshot(),
            },
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
        });
        let mut values = vec![agent_data, self.model.serialize(), self.env.serialize()];
        values.extend(extra_dicts.iter().cloned());
        values
            .into_iter()
            .fold(Value::Object(Map::new()), |base, value| {
                json_merge(base, value)
            })
    }

    pub fn save(&self, path: Option<&Path>, extra_dicts: &[Value]) -> Result<Value> {
        let data = self.serialize(extra_dicts);
        if let Some(path) = path {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(AgentError::other)?;
            }
            let serialized = serde_json::to_string_pretty(&data).map_err(AgentError::other)?;
            std::fs::write(path, serialized).map_err(|error| {
                AgentError::other(anyhow::anyhow!(
                    "could not write {}: {error}",
                    path.display()
                ))
            })?;
        }
        Ok(data)
    }
}

/// A small user-in-the-loop agent matching the Python `InteractiveAgent`
/// modes: `human`, `confirm`, and `yolo`.
pub struct InteractiveAgent {
    pub inner: DefaultAgent,
    pub mode: AgentMode,
    whitelist_actions: Vec<regex::Regex>,
    confirm_exit: bool,
}

impl InteractiveAgent {
    pub fn new(model: Box<dyn Model>, env: Box<dyn Environment>, config: AgentConfig) -> Self {
        let mode = config.mode;
        let confirm_exit = config.confirm_exit;
        let whitelist_actions = config
            .whitelist_actions
            .iter()
            .filter_map(|pattern| regex::Regex::new(pattern).ok())
            .collect();
        Self {
            inner: DefaultAgent::new(model, env, config),
            mode,
            whitelist_actions,
            confirm_exit,
        }
    }

    pub fn get_template_vars(&self, kwargs: Option<&Value>) -> Value {
        self.inner.get_template_vars(kwargs)
    }

    pub fn serialize(&self, extra_dicts: &[Value]) -> Value {
        self.inner.serialize(extra_dicts)
    }

    pub fn save(&self, path: Option<&Path>, extra_dicts: &[Value]) -> Result<Value> {
        self.inner.save(path, extra_dicts)
    }

    fn should_confirm(&self, action: &Action) -> bool {
        self.mode == AgentMode::Confirm
            && !self
                .whitelist_actions
                .iter()
                .any(|pattern| pattern.is_match(&action.command))
    }

    fn prompt(&self, text: &str) -> Result<String> {
        use std::io::Write;
        print!("{text}");
        std::io::stdout().flush().map_err(AgentError::other)?;
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .map_err(AgentError::other)?;
        Ok(input.trim_end().to_string())
    }

    pub async fn step(&mut self) -> Result<Vec<Message>> {
        let message = self.query().await?;
        self.execute_actions(&message).await
    }

    pub async fn query(&mut self) -> Result<Message> {
        self.inner.check_limits()?;
        if self.mode == AgentMode::Human {
            let command = self.prompt("> ")?;
            let mut message = Message::user(format!("User command: \n```bash\n{command}\n```"));
            message.set_extra("actions", serde_json::json!([{"command": command}]));
            let message = self.inner.add_messages(vec![message]).remove(0);
            return Ok(message);
        }
        self.inner.query().await
    }

    pub async fn execute_actions(&mut self, message: &Message) -> Result<Vec<Message>> {
        let actions = message.actions().map_err(AgentError::other)?;
        if actions.iter().any(|action| self.should_confirm(action)) {
            let answer = self.prompt(&format!(
                "Execute {} action(s)? [Enter] confirm, type comment to reject: ",
                actions.len()
            ))?;
            if !answer.trim().is_empty() {
                return Err(user_interrupt_message(
                    InterruptKind::UserRejection,
                    format!(
                        "Commands not executed. The user rejected your commands with the following message: {answer}"
                    ),
                )
                .into());
            }
        }
        match self.inner.execute_actions(message).await {
            Err(AgentError::Interrupt(interrupt))
                if interrupt.kind == InterruptKind::Submitted && self.confirm_exit =>
            {
                let answer =
                    self.prompt("Agent wants to finish. Type new task or Enter to quit: ")?;
                if answer.trim().is_empty() {
                    Err(AgentError::Interrupt(interrupt))
                } else {
                    Err(user_interrupt_message(
                        InterruptKind::UserNewTask,
                        format!("The user added a new task: {answer}"),
                    )
                    .into())
                }
            }
            result => result,
        }
    }
}

#[async_trait]
impl Agent for InteractiveAgent {
    async fn run(&mut self, task: &str, kwargs: Option<Value>) -> Result<Value> {
        self.inner
            .extra_template_vars
            .insert("task".to_string(), Value::String(task.to_string()));
        if let Some(Value::Object(kwargs)) = kwargs {
            for (key, value) in kwargs {
                self.inner.extra_template_vars.insert(key, value);
            }
        }

        self.inner.messages.clear();
        let system = self.inner.model.format_message(Message::system(
            self.inner
                .render_template(&self.inner.config.system_template, None)?,
        ))?;
        let instance = self.inner.model.format_message(Message::user(
            self.inner
                .render_template(&self.inner.config.instance_template, None)?,
        ))?;
        self.inner.add_messages(vec![system, instance]);

        loop {
            match self.step().await {
                Ok(_) => {
                    self.inner.n_consecutive_format_errors = 0;
                }
                Err(AgentError::Format(error)) => {
                    let billed_cost = error
                        .messages
                        .first()
                        .and_then(|message| message.extra_value("cost"))
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0);
                    self.inner.cost += billed_cost;
                    self.inner.n_consecutive_format_errors += 1;
                    if self.inner.config.max_consecutive_format_errors > 0
                        && self.inner.n_consecutive_format_errors
                            >= self.inner.config.max_consecutive_format_errors
                    {
                        let mut messages = error.messages;
                        messages.push(Message::exit_message(
                            "RepeatedFormatError",
                            "RepeatedFormatError",
                            "",
                        ));
                        self.inner.add_messages(messages);
                    } else {
                        self.inner.add_messages(error.messages);
                    }
                }
                Err(AgentError::Interrupt(interrupt)) => {
                    self.inner.add_messages(interrupt.messages);
                }
                Err(AgentError::Other(error)) => {
                    self.inner.handle_uncaught_exception(&error);
                    if self.inner.config.output_path.is_some() {
                        let output_path = self.inner.config.output_path.clone();
                        let _ = self.inner.save(output_path.as_deref(), &[]);
                    }
                    return Err(AgentError::Other(error));
                }
            }

            if self.inner.config.output_path.is_some() {
                let output_path = self.inner.config.output_path.clone();
                let _ = self.inner.save(output_path.as_deref(), &[]);
            }

            if self
                .inner
                .messages
                .last()
                .map(|message| message.role == "exit")
                .unwrap_or(false)
            {
                break;
            }
        }

        let extra = self
            .inner
            .messages
            .last()
            .and_then(|message| message.extra.as_ref())
            .map(|extra| Value::Object(extra.clone()))
            .unwrap_or(Value::Null);
        Ok(extra)
    }

    fn save(&self, path: Option<&Path>, extra_dicts: &[Value]) -> Result<Value> {
        InteractiveAgent::save(self, path, extra_dicts)
    }
}
#[async_trait]
impl Agent for DefaultAgent {
    async fn run(&mut self, task: &str, kwargs: Option<Value>) -> Result<Value> {
        self.extra_template_vars
            .insert("task".to_string(), Value::String(task.to_string()));
        if let Some(Value::Object(kwargs)) = kwargs {
            for (key, value) in kwargs {
                self.extra_template_vars.insert(key, value);
            }
        }

        self.messages.clear();
        let system = self.model.format_message(Message::system(
            self.render_template(&self.config.system_template, None)?,
        ))?;
        let instance = self.model.format_message(Message::user(
            self.render_template(&self.config.instance_template, None)?,
        ))?;
        self.add_messages(vec![system, instance]);

        loop {
            match self.step().await {
                Ok(_) => {
                    self.n_consecutive_format_errors = 0;
                }
                Err(AgentError::Format(error)) => {
                    let billed_cost = error
                        .messages
                        .first()
                        .and_then(|message| message.extra_value("cost"))
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0);
                    self.cost += billed_cost;
                    self.n_consecutive_format_errors += 1;
                    if self.config.max_consecutive_format_errors > 0
                        && self.n_consecutive_format_errors
                            >= self.config.max_consecutive_format_errors
                    {
                        let mut messages = error.messages;
                        messages.push(Message::exit_message(
                            "RepeatedFormatError",
                            "RepeatedFormatError",
                            "",
                        ));
                        self.add_messages(messages);
                    } else {
                        self.add_messages(error.messages);
                    }
                }
                Err(AgentError::Interrupt(interrupt)) => {
                    self.add_messages(interrupt.messages);
                }
                Err(AgentError::Other(error)) => {
                    self.handle_uncaught_exception(&error);
                    if self.config.output_path.is_some() {
                        let output_path = self.config.output_path.clone();
                        let _ = self.save(output_path.as_deref(), &[]);
                    }
                    return Err(AgentError::Other(error));
                }
            }

            // Save after every iteration so a crashed or interrupted run still
            // leaves a usable trajectory.
            if self.config.output_path.is_some() {
                let output_path = self.config.output_path.clone();
                let _ = self.save(output_path.as_deref(), &[]);
            }

            if self
                .messages
                .last()
                .map(|message| message.role == "exit")
                .unwrap_or(false)
            {
                break;
            }
        }

        let extra = self
            .messages
            .last()
            .and_then(|message| message.extra.as_ref())
            .map(|extra| Value::Object(extra.clone()))
            .unwrap_or(Value::Null);
        Ok(extra)
    }

    fn save(&self, path: Option<&Path>, extra_dicts: &[Value]) -> Result<Value> {
        DefaultAgent::save(self, path, extra_dicts)
    }
}

/// Build a user-visible interrupt message.
pub fn user_interrupt_message(kind: InterruptKind, content: impl Into<Value>) -> FlowInterrupt {
    FlowInterrupt::new(
        kind,
        vec![Message::new("user", content).with_extra({
            let mut extra = Map::new();
            extra.insert(
                "interrupt_type".to_string(),
                Value::String(kind.to_string()),
            );
            extra
        })],
    )
}
