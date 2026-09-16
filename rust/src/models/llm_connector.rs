//! Generic OpenAI-compatible model backed by `llm-connector`.
//!
//! `llm-connector` owns protocol/provider selection and HTTP transport.  This
//! module only maps mini-SWE-agent messages/actions to the connector's unified
//! types, and maps responses back to the internal message representation.

use std::time::Duration;

use anyhow::Context;
use async_trait::async_trait;
use llm_connector::types::{
    ChatRequest, ChatResponse, FunctionCall, Message as LlmMessage, Role, Tool, ToolCall,
};
use llm_connector::LlmClient;
use serde_json::{Map, Value};

use crate::{global_model_stats, Action, AgentError, Message, Model, Output, Result};

use super::{
    format_text_observation_messages, format_toolcall_observation_messages, parse_regex_actions,
    parse_toolcall_actions, timestamp_seconds, ModelConfig,
};

const DEFAULT_TEXT_ACTION_REGEX: &str = r"```mswea_bash_command\s*\n(.*?)\n```";

#[derive(Clone, Debug)]
pub enum ApiMode {
    ToolCalls,
    Text { action_regex: String },
}

/// Generic OpenAI-compatible model implementation.
pub struct LlmConnectorModel {
    pub config: ModelConfig,
    pub mode: ApiMode,
    client: LlmClient,
    effective_model_name: String,
    max_retries: u32,
}

impl LlmConnectorModel {
    pub fn from_value(value: Value) -> anyhow::Result<Self> {
        let config: ModelConfig =
            serde_json::from_value(value).context("invalid model configuration")?;
        let mode = if config.use_tool_calls {
            ApiMode::ToolCalls
        } else {
            ApiMode::Text {
                action_regex: config
                    .action_regex
                    .clone()
                    .unwrap_or_else(|| DEFAULT_TEXT_ACTION_REGEX.to_string()),
            }
        };
        Self::new(config, mode)
    }

    pub fn from_value_with_mode(value: Value, mode: ApiMode) -> anyhow::Result<Self> {
        let config: ModelConfig =
            serde_json::from_value(value).context("invalid model configuration")?;
        Self::new(config, mode)
    }

    pub fn new(mut config: ModelConfig, mode: ApiMode) -> anyhow::Result<Self> {
        config.model_name = config
            .resolved_model_name("MSWEA_MODEL_NAME")
            .context("model name is required")?;
        let base_url = resolve_base_url(&config);
        let api_key =
            resolve_api_key(&config.model_name, config.api_key.as_deref()).unwrap_or_default();
        let effective_model_name = effective_model_name(&config.model_name);
        let max_retries = config
            .max_retries
            .or_else(|| {
                std::env::var("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT")
                    .ok()
                    .and_then(|value| value.parse().ok())
            })
            .unwrap_or(10);

        let provider = config
            .provider
            .clone()
            .unwrap_or_else(|| infer_provider(&config.model_name));
        let service_name = config
            .service_name
            .clone()
            .unwrap_or_else(|| "openai_compatible".to_string());
        let client = create_client(&provider, &api_key, &base_url, &service_name)
            .map_err(|error| anyhow::anyhow!("could not create llm-connector client: {error}"))?;

        Ok(Self {
            config,
            mode,
            client,
            effective_model_name,
            max_retries,
        })
    }

    fn prepare_messages(&self, messages: &[Message]) -> Result<Vec<LlmMessage>> {
        messages.iter().map(to_llm_message).collect()
    }

    fn build_request(
        &self,
        messages: &[LlmMessage],
        kwargs: Option<&Value>,
    ) -> Result<ChatRequest> {
        let mut request = ChatRequest::new(self.effective_model_name.clone());
        request.messages = messages.to_vec();

        if matches!(self.mode, ApiMode::ToolCalls) {
            request.tools = Some(vec![bash_tool_definition()]);
        }

        apply_model_kwargs(&mut request, &self.config.model_kwargs)?;
        if let Some(Value::Object(kwargs)) = kwargs {
            apply_model_kwargs(&mut request, kwargs)?;
        }
        Ok(request)
    }

    async fn send_with_retry(&self, request: &ChatRequest) -> anyhow::Result<ChatResponse> {
        let mut attempt = 0u32;
        loop {
            attempt += 1;
            match self.client.chat(request).await {
                Ok(response) => return Ok(response),
                Err(error) if error.is_retryable() && attempt < self.max_retries => {
                    let delay = 1u64 << attempt.min(5);
                    tokio::time::sleep(Duration::from_secs(delay.min(30))).await;
                }
                Err(error) => return Err(error).context("llm-connector chat request failed"),
            }
        }
    }

    fn calculate_cost(&self, response: &ChatResponse) -> f64 {
        let (Some(input_price), Some(output_price)) = (
            self.config.input_cost_per_million,
            self.config.output_cost_per_million,
        ) else {
            return 0.0;
        };
        let input_tokens = response.prompt_tokens() as f64;
        let output_tokens = response.completion_tokens() as f64;
        (input_tokens * input_price + output_tokens * output_price) / 1_000_000.0
    }

    fn parse_actions(&self, response: &ChatResponse) -> Result<Vec<Action>> {
        let choice = response
            .choices
            .first()
            .ok_or_else(|| AgentError::other(anyhow::anyhow!("model response missing choices")))?;
        let finish_reason = choice.finish_reason.as_deref();
        match &self.mode {
            ApiMode::ToolCalls => {
                let tool_calls = choice.message.tool_calls.clone().unwrap_or_default();
                let tool_calls_value =
                    serde_json::to_value(&tool_calls).map_err(AgentError::other)?;
                let tool_calls_array = tool_calls_value.as_array().cloned().unwrap_or_default();
                parse_toolcall_actions(
                    &tool_calls_array,
                    &self.config.format_error_template,
                    finish_reason,
                )
            }
            ApiMode::Text { action_regex } => {
                let content = if response.content.is_empty() {
                    choice.message.content_as_text()
                } else {
                    response.content.clone()
                };
                parse_regex_actions(
                    &content,
                    action_regex,
                    &self.config.format_error_template,
                    finish_reason,
                )
            }
        }
    }

    async fn query_inner(&self, messages: &[Message], kwargs: Option<&Value>) -> Result<Message> {
        let prepared = self.prepare_messages(messages)?;
        let request = self.build_request(&prepared, kwargs)?;
        let response = self
            .send_with_retry(&request)
            .await
            .map_err(AgentError::other)?;
        let cost = self.calculate_cost(&response);
        global_model_stats()
            .lock()
            .map_err(|_| AgentError::other(anyhow::anyhow!("global model stats lock poisoned")))?
            .add(cost)
            .map_err(AgentError::other)?;

        let actions = match self.parse_actions(&response) {
            Ok(actions) => actions,
            Err(AgentError::Format(mut error)) => {
                if let Some(first) = error.messages.first_mut() {
                    first.set_extra("cost", serde_json::json!(cost));
                    first.set_extra(
                        "response",
                        serde_json::to_value(&response).unwrap_or(Value::Null),
                    );
                }
                return Err(AgentError::Format(error));
            }
            Err(error) => return Err(error),
        };

        let mut message = to_internal_message(&response)?;
        let mut extra = Map::new();
        extra.insert(
            "actions".to_string(),
            serde_json::to_value(&actions).map_err(AgentError::other)?,
        );
        extra.insert(
            "response".to_string(),
            serde_json::to_value(&response).unwrap_or(Value::Null),
        );
        extra.insert("cost".to_string(), serde_json::json!(cost));
        extra.insert(
            "timestamp".to_string(),
            serde_json::json!(timestamp_seconds()),
        );
        message.extra = Some(extra);
        Ok(message)
    }
}

#[async_trait]
impl Model for LlmConnectorModel {
    fn model_name(&self) -> &str {
        &self.config.model_name
    }

    async fn query(&self, messages: &[Message], kwargs: Option<Value>) -> Result<Message> {
        self.query_inner(messages, kwargs.as_ref()).await
    }

    fn format_observation_messages(
        &self,
        message: &Message,
        outputs: &[Output],
        template_vars: &Value,
    ) -> Result<Vec<Message>> {
        match &self.mode {
            ApiMode::ToolCalls => {
                let actions = message.actions().map_err(AgentError::other)?;
                format_toolcall_observation_messages(
                    &actions,
                    outputs,
                    &self.config.observation_template,
                    template_vars,
                )
            }
            ApiMode::Text { .. } => format_text_observation_messages(
                outputs,
                &self.config.observation_template,
                template_vars,
            ),
        }
    }

    fn get_template_vars(&self) -> Value {
        serde_json::to_value(&self.config).unwrap_or(Value::Null)
    }

    fn serialize(&self) -> Value {
        serde_json::json!({
            "info": {
                "config": {
                    "model": serde_json::to_value(&self.config).unwrap_or(Value::Null),
                    "model_type": "mini_swe_agent.models.llm_connector.LlmConnectorModel",
                }
            }
        })
    }
}

/// Text-based model adapter, matching the Python `litellm_textbased` class.
pub struct TextBasedModel(pub LlmConnectorModel);

impl TextBasedModel {
    pub fn from_value(value: Value) -> anyhow::Result<Self> {
        let mut config: ModelConfig =
            serde_json::from_value(value).context("invalid model configuration")?;
        config.use_tool_calls = false;
        let action_regex = config
            .action_regex
            .clone()
            .unwrap_or_else(|| DEFAULT_TEXT_ACTION_REGEX.to_string());
        Ok(Self(LlmConnectorModel::new(
            config,
            ApiMode::Text { action_regex },
        )?))
    }
}

#[async_trait]
impl Model for TextBasedModel {
    fn model_name(&self) -> &str {
        self.0.model_name()
    }

    async fn query(&self, messages: &[Message], kwargs: Option<Value>) -> Result<Message> {
        self.0.query(messages, kwargs).await
    }

    fn format_message(&self, message: Message) -> Result<Message> {
        self.0.format_message(message)
    }

    fn format_observation_messages(
        &self,
        message: &Message,
        outputs: &[Output],
        template_vars: &Value,
    ) -> Result<Vec<Message>> {
        self.0
            .format_observation_messages(message, outputs, template_vars)
    }

    fn get_template_vars(&self) -> Value {
        self.0.get_template_vars()
    }

    fn serialize(&self) -> Value {
        let mut value = self.0.serialize();
        if let Some(config) = value.pointer_mut("/info/config/model_type") {
            *config =
                Value::String("mini_swe_agent.models.llm_connector.TextBasedModel".to_string());
        }
        value
    }
}

fn bash_tool_definition() -> Tool {
    Tool::function(
        "bash",
        Some("Execute a bash command".to_string()),
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                }
            },
            "required": ["command"]
        }),
    )
}

fn to_llm_message(message: &Message) -> Result<LlmMessage> {
    let role = match message.role.as_str() {
        "system" => Role::System,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        _ => Role::User,
    };
    let mut llm_message = LlmMessage::text(role, content_as_text(message));
    if let Some(tool_calls) = message.fields.get("tool_calls") {
        llm_message.tool_calls = Some(parse_tool_calls(tool_calls)?);
    }
    if let Some(tool_call_id) = message.fields.get("tool_call_id").and_then(Value::as_str) {
        llm_message.tool_call_id = Some(tool_call_id.to_string());
    }
    for key in ["reasoning_content", "reasoning", "thinking", "thought"] {
        if let Some(value) = message.fields.get(key).and_then(Value::as_str) {
            match key {
                "reasoning_content" => llm_message.reasoning_content = Some(value.to_string()),
                "reasoning" => llm_message.reasoning = Some(value.to_string()),
                "thinking" => llm_message.thinking = Some(value.to_string()),
                "thought" => llm_message.thought = Some(value.to_string()),
                _ => {}
            }
        }
    }
    Ok(llm_message)
}

fn to_internal_message(response: &ChatResponse) -> Result<Message> {
    let choice = response
        .choices
        .first()
        .ok_or_else(|| AgentError::other(anyhow::anyhow!("model response missing choices")))?;
    let llm_message = &choice.message;
    let role = match llm_message.role {
        Role::System => "system",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
        Role::User => "user",
    };
    let mut message = Message::new(role, llm_message.content_as_text());
    if let Some(tool_calls) = &llm_message.tool_calls {
        message.fields.insert(
            "tool_calls".to_string(),
            serde_json::to_value(tool_calls).map_err(AgentError::other)?,
        );
    }
    if let Some(tool_call_id) = &llm_message.tool_call_id {
        message.fields.insert(
            "tool_call_id".to_string(),
            Value::String(tool_call_id.clone()),
        );
    }
    for (key, value) in [
        ("reasoning_content", &llm_message.reasoning_content),
        ("reasoning", &llm_message.reasoning),
        ("thinking", &llm_message.thinking),
        ("thought", &llm_message.thought),
    ] {
        if let Some(value) = value {
            message
                .fields
                .insert(key.to_string(), Value::String(value.clone()));
        }
    }
    Ok(message)
}

fn parse_tool_calls(value: &Value) -> Result<Vec<ToolCall>> {
    let array = value
        .as_array()
        .ok_or_else(|| AgentError::other(anyhow::anyhow!("tool_calls must be an array")))?;
    let mut tool_calls = Vec::with_capacity(array.len());
    for item in array {
        let function = item.get("function").unwrap_or(&Value::Null);
        tool_calls.push(ToolCall {
            id: item
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            call_type: item
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("function")
                .to_string(),
            function: FunctionCall {
                name: function
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                arguments: function
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("{}")
                    .to_string(),
                thought_signature: None,
            },
            index: None,
            thought_signature: None,
        });
    }
    Ok(tool_calls)
}

fn content_as_text(message: &Message) -> String {
    match &message.content {
        Value::String(value) => value.clone(),
        Value::Null => String::new(),
        Value::Array(items) => items
            .iter()
            .filter_map(|item| {
                item.get("text")
                    .or_else(|| item.get("content"))
                    .and_then(Value::as_str)
            })
            .collect::<Vec<_>>()
            .join("\n"),
        value => value.to_string(),
    }
}

fn apply_model_kwargs(request: &mut ChatRequest, kwargs: &Map<String, Value>) -> Result<()> {
    if let Some(value) = kwargs.get("temperature").and_then(Value::as_f64) {
        request.temperature = Some(value as f32);
    }
    if let Some(value) = kwargs.get("top_p").and_then(Value::as_f64) {
        request.top_p = Some(value as f32);
    }
    if let Some(value) = kwargs.get("max_tokens").and_then(Value::as_u64) {
        request.max_tokens = Some(value as u32);
    }
    if let Some(value) = kwargs.get("seed").and_then(Value::as_u64) {
        request.seed = Some(value);
    }
    if let Some(value) = kwargs.get("presence_penalty").and_then(Value::as_f64) {
        request.presence_penalty = Some(value as f32);
    }
    if let Some(value) = kwargs.get("frequency_penalty").and_then(Value::as_f64) {
        request.frequency_penalty = Some(value as f32);
    }
    if let Some(value) = kwargs.get("user").and_then(Value::as_str) {
        request.user = Some(value.to_string());
    }
    if let Some(values) = kwargs.get("stop").and_then(Value::as_array) {
        request.stop = Some(
            values
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect(),
        );
    }
    Ok(())
}

fn resolve_base_url(config: &ModelConfig) -> String {
    if let Some(base_url) = &config.base_url {
        return base_url.clone();
    }
    for key in ["OPENAI_BASE_URL", "OPENAI_API_BASE", "LITELLM_API_BASE"] {
        if let Ok(value) = std::env::var(key) {
            if !value.trim().is_empty() {
                return value;
            }
        }
    }
    if config.model_name.starts_with("openrouter/") {
        return "https://openrouter.ai/api/v1".to_string();
    }
    "https://api.openai.com/v1".to_string()
}

fn resolve_api_key(model_name: &str, explicit: Option<&str>) -> Option<String> {
    if let Some(explicit) = explicit {
        if !explicit.trim().is_empty() {
            return Some(explicit.to_string());
        }
    }
    let keys: &[&str] = if model_name.starts_with("openrouter/") {
        &["OPENROUTER_API_KEY", "OPENAI_API_KEY", "LITELLM_API_KEY"]
    } else if model_name.starts_with("anthropic/") {
        &["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LITELLM_API_KEY"]
    } else {
        &["OPENAI_API_KEY", "LITELLM_API_KEY"]
    };
    keys.iter()
        .find_map(|key| std::env::var(key).ok().filter(|value| !value.is_empty()))
}

fn create_client(
    provider: &str,
    api_key: &str,
    base_url: &str,
    service_name: &str,
) -> std::result::Result<LlmClient, llm_connector::LlmConnectorError> {
    match provider.to_ascii_lowercase().as_str() {
        "anthropic" | "claude" => LlmClient::anthropic(api_key, base_url),
        "google" | "gemini" => LlmClient::google(api_key, base_url),
        "ollama" => LlmClient::ollama(base_url),
        "zhipu" | "glm" => LlmClient::zhipu(api_key, base_url),
        "aliyun" | "qwen" | "dashscope" => LlmClient::aliyun(api_key, base_url),
        "openai_compatible" | "compatible" => {
            LlmClient::openai_compatible(api_key, base_url, service_name)
        }
        _ => LlmClient::openai(api_key, base_url),
    }
}

fn infer_provider(model_name: &str) -> String {
    let provider = model_name
        .split_once('/')
        .map(|(provider, _)| provider)
        .unwrap_or(model_name);
    match provider.to_ascii_lowercase().as_str() {
        "anthropic" | "claude" => "anthropic".to_string(),
        "google" | "gemini" => "google".to_string(),
        "ollama" => "ollama".to_string(),
        "zhipu" | "glm" => "zhipu".to_string(),
        "aliyun" | "qwen" | "dashscope" => "aliyun".to_string(),
        "deepseek" | "moonshot" | "openrouter" => "openai_compatible".to_string(),
        _ => "openai".to_string(),
    }
}

fn effective_model_name(model_name: &str) -> String {
    for prefix in [
        "openai/",
        "openrouter/",
        "anthropic/",
        "google/",
        "gemini/",
        "ollama/",
        "zhipu/",
        "glm/",
        "aliyun/",
        "qwen/",
        "dashscope/",
        "deepseek/",
        "moonshot/",
    ] {
        if let Some(stripped) = model_name.strip_prefix(prefix) {
            return stripped.to_string();
        }
    }
    model_name.to_string()
}
