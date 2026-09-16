use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use serde_json::{json, Map, Value};

use mini_swe_agent::{
    config::{get_config_from_spec, recursive_merge},
    Action, Agent, AgentConfig, AgentMode, DefaultAgent, Environment, InteractiveAgent, Message,
    Model, Output, Result,
};

struct MockModel {
    calls: AtomicUsize,
}

#[async_trait]
impl Model for MockModel {
    fn model_name(&self) -> &str {
        "mock"
    }

    async fn query(&self, _messages: &[Message], _kwargs: Option<Value>) -> Result<Message> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        if call == 0 {
            let mut message = Message::assistant("calling bash");
            message.set_extra("actions", json!([{"command": "echo hi"}]));
            message.set_extra("cost", json!(0.0));
            Ok(message)
        } else {
            Ok(Message::exit_message("Submitted", "done", "done"))
        }
    }

    fn format_observation_messages(
        &self,
        _message: &Message,
        outputs: &[Output],
        _template_vars: &Value,
    ) -> Result<Vec<Message>> {
        Ok(outputs
            .iter()
            .map(|output| Message::user(output.output.clone()))
            .collect())
    }

    fn get_template_vars(&self) -> Value {
        json!({})
    }

    fn serialize(&self) -> Value {
        json!({"info": {"config": {"model": {"model_name": "mock"}}}})
    }
}

struct MockEnvironment;

#[async_trait]
impl Environment for MockEnvironment {
    async fn execute(
        &self,
        action: &Action,
        _cwd: Option<&str>,
        _timeout: Option<u64>,
    ) -> Result<Output> {
        assert_eq!(action.command, "echo hi");
        Ok(Output::success("hi\n", 0))
    }

    fn get_template_vars(&self) -> Value {
        json!({})
    }

    fn serialize(&self) -> Value {
        json!({"info": {"config": {"environment": {"type": "mock"}}}})
    }
}

#[test]
fn merges_configs_recursively() {
    let merged = recursive_merge([
        json!({"model": {"model_kwargs": {"temperature": 0.0}}, "agent": {"step_limit": 1}}),
        json!({"model": {"model_kwargs": {"top_p": 0.9, "temperature": 0.7}}}),
    ]);
    assert_eq!(merged["model"]["model_kwargs"]["temperature"], 0.7);
    assert_eq!(merged["model"]["model_kwargs"]["top_p"], 0.9);
    assert_eq!(merged["agent"]["step_limit"], 1);
}

#[test]
fn loads_builtin_mini_config() {
    let config = get_config_from_spec("mini.yaml").unwrap();
    assert!(config["agent"]["system_template"]
        .as_str()
        .unwrap()
        .contains("helpful"));
    assert_eq!(config["model"]["model_kwargs"]["drop_params"], true);
}

#[tokio::test]
async fn default_agent_completes_linear_flow() {
    let model = MockModel {
        calls: AtomicUsize::new(0),
    };
    let env = MockEnvironment;
    let config = AgentConfig {
        system_template: "system".to_string(),
        instance_template: "task: {{task}}".to_string(),
        cost_limit: 0.0,
        ..AgentConfig::default()
    };
    let mut agent = DefaultAgent::new(Box::new(model), Box::new(env), config);
    let result = agent.run("demo", None).await.unwrap();
    assert_eq!(result["exit_status"], "Submitted");
    assert_eq!(result["submission"], "done");
    assert_eq!(agent.messages.len(), 5);
}

#[tokio::test]
async fn interactive_agent_yolo_completes_without_prompt() {
    let model = MockModel {
        calls: AtomicUsize::new(0),
    };
    let env = MockEnvironment;
    let config = AgentConfig {
        system_template: "system".to_string(),
        instance_template: "task: {{task}}".to_string(),
        cost_limit: 0.0,
        mode: AgentMode::Yolo,
        ..AgentConfig::default()
    };
    let mut agent = InteractiveAgent::new(Box::new(model), Box::new(env), config);
    let result = agent.run("demo", None).await.unwrap();
    assert_eq!(result["exit_status"], "Submitted");
}
#[tokio::test]
async fn local_environment_executes_commands() {
    use mini_swe_agent::LocalEnvironment;
    let env = LocalEnvironment::new(Default::default());
    let output = env
        .execute(&Action::new("echo hello"), None, None)
        .await
        .unwrap();
    assert_eq!(output.returncode, 0);
    assert!(output.output.trim() == "hello");
}

#[tokio::test]
async fn local_environment_detects_submission() {
    use mini_swe_agent::LocalEnvironment;
    let env = LocalEnvironment::new(Default::default());
    let error = env
        .execute(
            &Action::new("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo final answer"),
            None,
            None,
        )
        .await
        .unwrap_err();
    let mini_swe_agent::AgentError::Interrupt(interrupt) = error else {
        panic!("expected submission interrupt");
    };
    assert_eq!(interrupt.kind, mini_swe_agent::InterruptKind::Submitted);
    let extra: &Map<String, Value> = interrupt.messages[0].extra.as_ref().unwrap();
    assert_eq!(extra["submission"].as_str().unwrap().trim(), "final answer");
}
