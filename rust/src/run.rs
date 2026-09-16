//! `mini` command-line entry point.

use std::io::Read;

use anyhow::{bail, Context};
use clap::Parser;
use serde_json::{Map, Value};

use crate::{
    config::{get_config_from_spec, global_config_dir, load_global_config, recursive_merge},
    models::{ApiMode, LlmConnectorModel, TextBasedModel},
    Agent, AgentConfig, DefaultAgent, DockerEnvironment, InteractiveAgent, LocalEnvironment,
};

#[derive(Debug, Parser)]
#[command(
    name = "mini",
    version,
    about = "Run mini-SWE-agent in your local environment"
)]
pub struct Cli {
    #[arg(short = 'm', long = "model")]
    pub model: Option<String>,

    #[arg(long = "model-class")]
    pub model_class: Option<String>,

    #[arg(long = "agent-class")]
    pub agent_class: Option<String>,

    #[arg(long = "environment-class")]
    pub environment_class: Option<String>,

    #[arg(short = 't', long = "task")]
    pub task: Option<String>,

    #[arg(short = 'y', long = "yolo")]
    pub yolo: bool,

    #[arg(short = 'l', long = "cost-limit")]
    pub cost_limit: Option<f64>,

    #[arg(short = 'c', long = "config", default_value = "mini.yaml")]
    pub config_spec: Vec<String>,

    #[arg(short = 'o', long = "output")]
    pub output: Option<String>,

    #[arg(long = "exit-immediately")]
    pub exit_immediately: bool,
}

pub async fn run(cli: Cli) -> anyhow::Result<()> {
    init_tracing();
    load_global_config();

    let mut configs = cli
        .config_spec
        .iter()
        .map(|spec| get_config_from_spec(spec))
        .collect::<anyhow::Result<Vec<_>>>()?;
    configs.push(cli_overrides(&cli));

    let config = recursive_merge(configs);
    let task = resolve_task(&config, &cli)?;

    let model = build_model(&config)?;
    let environment = build_environment(&config).await?;
    let agent_config = build_agent_config(&config, &cli)?;
    let mut agent = build_agent(&config, model, environment, agent_config);
    agent.run(&task, None).await?;

    if let Some(path) = config
        .get("agent")
        .and_then(|agent| agent.get("output_path"))
        .and_then(Value::as_str)
    {
        eprintln!("Saved trajectory to '{path}'");
    }
    Ok(())
}

fn cli_overrides(cli: &Cli) -> Value {
    let mut root = Map::new();
    let mut run = Map::new();
    let mut agent = Map::new();
    let mut model = Map::new();
    let mut environment = Map::new();

    if let Some(task) = &cli.task {
        run.insert("task".to_string(), Value::String(task.clone()));
    }
    if let Some(agent_class) = &cli.agent_class {
        agent.insert(
            "agent_class".to_string(),
            Value::String(agent_class.clone()),
        );
    }
    if cli.yolo {
        agent.insert("mode".to_string(), Value::String("yolo".to_string()));
    }
    if let Some(cost_limit) = cli.cost_limit {
        agent.insert("cost_limit".to_string(), serde_json::json!(cost_limit));
    }
    if cli.exit_immediately {
        agent.insert("confirm_exit".to_string(), Value::Bool(false));
    }
    if let Some(output) = &cli.output {
        agent.insert("output_path".to_string(), Value::String(output.clone()));
    } else if !agent.is_empty() {
        let default_output = global_config_dir().join("last_mini_run.traj.json");
        agent.insert(
            "output_path".to_string(),
            Value::String(default_output.to_string_lossy().into_owned()),
        );
    }

    if let Some(model_class) = &cli.model_class {
        model.insert(
            "model_class".to_string(),
            Value::String(model_class.clone()),
        );
    }
    if let Some(model_name) = &cli.model {
        model.insert("model_name".to_string(), Value::String(model_name.clone()));
    }
    if let Some(environment_class) = &cli.environment_class {
        environment.insert(
            "environment_class".to_string(),
            Value::String(environment_class.clone()),
        );
    }

    if !run.is_empty() {
        root.insert("run".to_string(), Value::Object(run));
    }
    if !agent.is_empty() {
        root.insert("agent".to_string(), Value::Object(agent));
    }
    if !model.is_empty() {
        root.insert("model".to_string(), Value::Object(model));
    }
    if !environment.is_empty() {
        root.insert("environment".to_string(), Value::Object(environment));
    }
    Value::Object(root)
}

fn resolve_task(config: &Value, cli: &Cli) -> anyhow::Result<String> {
    if let Some(task) = cli.task.clone().or_else(|| {
        config
            .get("run")
            .and_then(|run| run.get("task"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
    }) {
        if !task.trim().is_empty() {
            return Ok(task);
        }
    }
    eprintln!("What do you want to do?");
    let mut task = String::new();
    std::io::stdin()
        .read_to_string(&mut task)
        .context("could not read task from stdin")?;
    if task.trim().is_empty() {
        bail!("no task provided");
    }
    Ok(task)
}

fn build_model(config: &Value) -> anyhow::Result<Box<dyn crate::Model>> {
    let model_config = config
        .get("model")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    let model_class = model_config
        .get("model_class")
        .and_then(Value::as_str)
        .unwrap_or("");
    let lower = model_class.to_ascii_lowercase();
    if lower.contains("response") {
        bail!("response-API models are not yet implemented in the Rust runtime");
    }
    if lower.contains("textbased") || lower.contains("text_based") {
        Ok(Box::new(TextBasedModel::from_value(model_config)?))
    } else {
        Ok(Box::new(LlmConnectorModel::from_value_with_mode(
            model_config,
            ApiMode::ToolCalls,
        )?))
    }
}

fn build_agent(
    config: &Value,
    model: Box<dyn crate::Model>,
    environment: Box<dyn crate::Environment>,
    agent_config: AgentConfig,
) -> Box<dyn Agent> {
    let agent_class = config
        .get("agent")
        .and_then(|agent| agent.get("agent_class"))
        .and_then(Value::as_str)
        .unwrap_or("interactive")
        .to_ascii_lowercase();
    if agent_class.contains("default") {
        Box::new(DefaultAgent::new(model, environment, agent_config))
    } else {
        Box::new(InteractiveAgent::new(model, environment, agent_config))
    }
}

async fn build_environment(config: &Value) -> anyhow::Result<Box<dyn crate::Environment>> {
    let environment_config = config
        .get("environment")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    let environment_class = environment_config
        .get("environment_class")
        .and_then(Value::as_str)
        .unwrap_or("local")
        .to_ascii_lowercase();
    if environment_class.contains("docker") {
        Ok(Box::new(
            DockerEnvironment::from_value(environment_config).await?,
        ))
    } else {
        Ok(Box::new(LocalEnvironment::from_value(environment_config)?))
    }
}

fn build_agent_config(config: &Value, cli: &Cli) -> anyhow::Result<AgentConfig> {
    let agent_value = config
        .get("agent")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    let mut agent_config: AgentConfig =
        serde_json::from_value(agent_value).context("invalid agent configuration")?;
    if let Some(cost_limit) = cli.cost_limit {
        agent_config.cost_limit = cost_limit;
    }
    if let Some(output) = &cli.output {
        agent_config.output_path = Some(output.into());
    } else if agent_config.output_path.is_none() {
        agent_config.output_path = Some(global_config_dir().join("last_mini_run.traj.json"));
    }
    Ok(agent_config)
}

fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}
