use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use async_trait::async_trait;
use serde_json::{json, Value};

use mini_swe_agent::{
    Action, Agent, AgentConfig, DefaultAgent, Environment, Message, Model, Output, Result,
};

struct FastModel {
    steps: usize,
    calls: AtomicUsize,
}

#[async_trait]
impl Model for FastModel {
    fn model_name(&self) -> &str {
        "fast"
    }

    async fn query(&self, _messages: &[Message], _kwargs: Option<Value>) -> Result<Message> {
        if self.calls.fetch_add(1, Ordering::Relaxed) >= self.steps {
            Ok(Message::exit_message("Submitted", "", ""))
        } else {
            let mut message = Message::assistant("x");
            message.set_extra("actions", json!([{"command": "echo x"}]));
            message.set_extra("cost", json!(0.0));
            Ok(message)
        }
    }

    fn format_observation_messages(
        &self,
        _message: &Message,
        outputs: &[Output],
        _template_vars: &Value,
    ) -> Result<Vec<Message>> {
        Ok(vec![Message::user("observed"); outputs.len()])
    }

    fn get_template_vars(&self) -> Value {
        json!({})
    }

    fn serialize(&self) -> Value {
        json!({})
    }
}

struct FastEnv;

#[async_trait]
impl Environment for FastEnv {
    async fn execute(
        &self,
        _action: &Action,
        _cwd: Option<&str>,
        _timeout: Option<u64>,
    ) -> Result<Output> {
        Ok(Output::success("ok", 0))
    }

    fn get_template_vars(&self) -> Value {
        json!({})
    }

    fn serialize(&self) -> Value {
        json!({})
    }
}

#[tokio::main]
async fn main() {
    let steps: usize = std::env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(2000);
    let model = FastModel {
        steps,
        calls: AtomicUsize::new(0),
    };
    let config = AgentConfig {
        system_template: "system".to_string(),
        instance_template: "{{task}}".to_string(),
        step_limit: 0,
        cost_limit: 0.0,
        ..AgentConfig::default()
    };
    let mut agent = DefaultAgent::new(Box::new(model), Box::new(FastEnv), config);
    let start = Instant::now();
    let _ = agent.run("bench", None).await;
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "{{\"steps\": {steps}, \"messages\": {}, \"seconds\": {elapsed}, \"steps_per_second\": {}}}",
        agent.messages.len(),
        steps as f64 / elapsed
    );
}
