//! Model implementations and protocol helpers.

pub mod llm_connector;

pub use llm_connector::{ApiMode, LlmConnectorModel, TextBasedModel};

use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{Action, AgentError, FormatError, Message, Result};

pub const BASH_TOOL_NAME: &str = "bash";

pub fn bash_tool_definition() -> Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": BASH_TOOL_NAME,
            "description": "Execute a bash command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute"
                    }
                },
                "required": ["command"]
            }
        }
    })
}

fn default_observation_template() -> String {
    "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>".to_string()
}

fn default_format_error_template() -> String {
    "{{ error }}".to_string()
}

/// Shared configuration for OpenAI-compatible chat-completions models.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    pub model_name: String,
    pub model_class: Option<String>,
    pub provider: Option<String>,
    pub service_name: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub model_kwargs: Map<String, Value>,
    pub observation_template: String,
    pub format_error_template: String,
    pub multimodal_regex: String,
    pub action_regex: Option<String>,
    pub set_cache_control: Option<String>,
    pub cost_tracking: String,
    pub max_retries: Option<u32>,
    pub request_timeout_secs: u64,
    pub input_cost_per_million: Option<f64>,
    pub output_cost_per_million: Option<f64>,
    pub extra_headers: Map<String, Value>,
    pub use_tool_calls: bool,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_name: String::new(),
            model_class: None,
            provider: None,
            service_name: None,
            base_url: None,
            api_key: None,
            model_kwargs: Map::new(),
            observation_template: default_observation_template(),
            format_error_template: default_format_error_template(),
            multimodal_regex: String::new(),
            action_regex: None,
            set_cache_control: None,
            cost_tracking: "default".to_string(),
            max_retries: None,
            request_timeout_secs: 120,
            input_cost_per_million: None,
            output_cost_per_million: None,
            extra_headers: Map::new(),
            use_tool_calls: true,
            extra: Map::new(),
        }
    }
}

impl ModelConfig {
    pub fn resolved_model_name(&self, fallback_env: &str) -> anyhow::Result<String> {
        if !self.model_name.is_empty() {
            return Ok(self.model_name.clone());
        }
        std::env::var(fallback_env)
            .ok()
            .filter(|value| !value.trim().is_empty())
            .context("no model set; pass --model or set MSWEA_MODEL_NAME")
    }
}

pub fn timestamp_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or_default()
}

fn format_error_message(
    error: impl Into<String>,
    actions: &[Action],
    has_tool_calls: bool,
    format_error_template: &str,
    finish_reason: Option<&str>,
) -> Message {
    let context = serde_json::json!({
        "error": error.into(),
        "actions": actions.iter().map(|action| action.command.clone()).collect::<Vec<_>>(),
        "has_tool_calls": has_tool_calls,
        "finish_reason": finish_reason,
    });
    let content = crate::template::render(format_error_template, &context)
        .unwrap_or_else(|_| "Format error".to_string());
    let mut message = Message::user(content);
    message.set_extra("interrupt_type", Value::String("FormatError".to_string()));
    message
}

/// Parse OpenAI-style tool calls into shell actions.
pub fn parse_toolcall_actions(
    tool_calls: &[Value],
    format_error_template: &str,
    finish_reason: Option<&str>,
) -> Result<Vec<Action>> {
    if tool_calls.is_empty() {
        return Err(FormatError::new(vec![format_error_message(
            "No tool calls found in the response. Every response MUST include at least one tool call.",
            &[],
            false,
            format_error_template,
            finish_reason,
        )])
        .into());
    }

    let mut actions = Vec::with_capacity(tool_calls.len());
    for tool_call in tool_calls {
        let null_function = Value::Null;
        let function = tool_call.get("function").unwrap_or(&null_function);
        let name = function.get("name").and_then(Value::as_str).unwrap_or("");
        let arguments = function
            .get("arguments")
            .and_then(Value::as_str)
            .unwrap_or("{}");
        let id = tool_call.get("id").and_then(Value::as_str).unwrap_or("");

        let mut error = String::new();
        let parsed: Value = match serde_json::from_str(arguments) {
            Ok(value) => value,
            Err(parse_error) => {
                error.push_str(&format!(
                    "Error parsing tool call arguments: {parse_error}."
                ));
                Value::Null
            }
        };
        if name != BASH_TOOL_NAME {
            error.push_str(&format!("Unknown tool '{name}'."));
        }
        let command = parsed
            .get("command")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        if command.is_none() {
            error.push_str("Missing 'command' argument in bash tool call.");
        }
        if !error.is_empty() {
            return Err(FormatError::new(vec![format_error_message(
                error.trim(),
                &[],
                true,
                format_error_template,
                finish_reason,
            )])
            .into());
        }
        actions.push(Action::with_tool_call_id(command.unwrap_or_default(), id));
    }
    Ok(actions)
}

/// Parse exactly one shell action from free-form text.
pub fn parse_regex_actions(
    content: &str,
    action_regex: &str,
    format_error_template: &str,
    finish_reason: Option<&str>,
) -> Result<Vec<Action>> {
    let regex = regex::Regex::new(action_regex)
        .map_err(|error| AgentError::other(anyhow::anyhow!("invalid action_regex: {error}")))?;
    let actions: Vec<Action> = regex
        .captures_iter(content)
        .filter_map(|captures| captures.get(1).map(|mat| Action::new(mat.as_str().trim())))
        .collect();
    if actions.len() != 1 {
        let error = format!("Expected exactly 1 action, found {}.", actions.len());
        return Err(FormatError::new(vec![format_error_message(
            error,
            &actions,
            false,
            format_error_template,
            finish_reason,
        )])
        .into());
    }
    Ok(actions)
}

/// Render observation messages for tool-call based models.
pub fn format_toolcall_observation_messages(
    actions: &[Action],
    outputs: &[crate::Output],
    observation_template: &str,
    template_vars: &Value,
) -> Result<Vec<Message>> {
    let not_executed = crate::Output::success("", -1);
    let mut messages = Vec::with_capacity(actions.len().max(outputs.len()));
    for (index, action) in actions.iter().enumerate() {
        let output = outputs.get(index).unwrap_or(&not_executed);
        let output_value = serde_json::to_value(output).map_err(AgentError::other)?;
        let content = crate::template::render_with_output(
            observation_template,
            template_vars,
            &output_value,
        )?;
        let mut extra = Map::new();
        extra.insert(
            "raw_output".to_string(),
            Value::String(output.output.clone()),
        );
        extra.insert(
            "returncode".to_string(),
            Value::Number(output.returncode.into()),
        );
        extra.insert(
            "timestamp".to_string(),
            serde_json::json!(timestamp_seconds()),
        );
        extra.insert(
            "exception_info".to_string(),
            Value::String(output.exception_info.clone()),
        );
        for (key, value) in &output.extra {
            extra.insert(key.clone(), value.clone());
        }
        let mut message = Message::new(
            if action.tool_call_id.is_some() {
                "tool"
            } else {
                "user"
            },
            content,
        )
        .with_extra(extra);
        if let Some(tool_call_id) = &action.tool_call_id {
            message.fields.insert(
                "tool_call_id".to_string(),
                Value::String(tool_call_id.clone()),
            );
        }
        messages.push(message);
    }
    Ok(messages)
}

/// Render observation messages for free-form text models.
pub fn format_text_observation_messages(
    outputs: &[crate::Output],
    observation_template: &str,
    template_vars: &Value,
) -> Result<Vec<Message>> {
    let mut messages = Vec::with_capacity(outputs.len());
    for output in outputs {
        let output_value = serde_json::to_value(output).map_err(AgentError::other)?;
        let content = crate::template::render_with_output(
            observation_template,
            template_vars,
            &output_value,
        )?;
        let mut extra = Map::new();
        extra.insert(
            "raw_output".to_string(),
            Value::String(output.output.clone()),
        );
        extra.insert(
            "returncode".to_string(),
            Value::Number(output.returncode.into()),
        );
        extra.insert(
            "timestamp".to_string(),
            serde_json::json!(timestamp_seconds()),
        );
        extra.insert(
            "exception_info".to_string(),
            Value::String(output.exception_info.clone()),
        );
        for (key, value) in &output.extra {
            extra.insert(key.clone(), value.clone());
        }
        messages.push(Message::user(content).with_extra(extra));
    }
    Ok(messages)
}
