//! Docker/Podman execution environment using direct CLI calls.

use std::process::Stdio;
use std::time::Duration;

use anyhow::{bail, Context};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use uuid::Uuid;

use crate::{Action, Environment, FlowInterrupt, Output, Result};

use super::{config_to_value, merge_environment_template_vars, normalize_newlines};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct DockerEnvironmentConfig {
    pub image: String,
    pub cwd: String,
    pub env: Map<String, Value>,
    pub forward_env: Vec<String>,
    pub timeout: u64,
    pub executable: String,
    pub run_args: Vec<String>,
    pub container_timeout: String,
    pub pull_timeout: u64,
    pub interpreter: Vec<String>,
}

impl Default for DockerEnvironmentConfig {
    fn default() -> Self {
        Self {
            image: String::new(),
            cwd: "/".to_string(),
            env: Map::new(),
            forward_env: Vec::new(),
            timeout: 30,
            executable: std::env::var("MSWEA_DOCKER_EXECUTABLE")
                .unwrap_or_else(|_| "docker".to_string()),
            run_args: vec!["--rm".to_string()],
            container_timeout: "2h".to_string(),
            pull_timeout: 120,
            interpreter: vec!["bash".to_string(), "-lc".to_string()],
        }
    }
}

pub struct DockerEnvironment {
    pub config: DockerEnvironmentConfig,
    pub container_id: String,
    pub container_name: String,
}

impl DockerEnvironment {
    pub async fn new(config: DockerEnvironmentConfig) -> anyhow::Result<Self> {
        if config.image.trim().is_empty() {
            bail!("docker environment requires an image");
        }
        let container_name = format!("minisweagent-{}", &Uuid::new_v4().simple().to_string()[..8]);
        let mut command = Command::new(&config.executable);
        command
            .arg("run")
            .arg("-d")
            .arg("--name")
            .arg(&container_name);
        command.arg("-w").arg(&config.cwd);
        command.args(&config.run_args);
        command
            .arg(&config.image)
            .arg("sleep")
            .arg(&config.container_timeout);
        let (stdout, returncode) = run_captured(&mut command, config.pull_timeout).await?;
        if returncode != 0 {
            bail!("could not start docker container: {stdout}");
        }
        let container_id = stdout.trim().to_string();
        if container_id.is_empty() {
            bail!("docker run returned an empty container id");
        }
        Ok(Self {
            config,
            container_id,
            container_name,
        })
    }

    pub async fn from_value(value: Value) -> anyhow::Result<Self> {
        let config: DockerEnvironmentConfig =
            serde_json::from_value(value).context("invalid docker environment config")?;
        Self::new(config).await
    }

    async fn execute_command(&self, mut command: Command, timeout: u64) -> Result<Output> {
        match run_captured(&mut command, timeout).await {
            Ok((output, returncode)) => Ok(Output::success(output, returncode)),
            Err(error) => Ok(Output::failure(error, "RuntimeError", "")),
        }
    }

    fn check_finished(&self, output: &Output) -> Result<()> {
        let trimmed = output.output.trim_start();
        let (first_line, submission) = match trimmed.split_once('\n') {
            Some((first, rest)) => (first, rest),
            None => (trimmed, ""),
        };
        if first_line.trim() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" && output.returncode == 0 {
            return Err(FlowInterrupt::submitted(submission).into());
        }
        Ok(())
    }
}

#[async_trait]
impl Environment for DockerEnvironment {
    async fn execute(
        &self,
        action: &Action,
        cwd: Option<&str>,
        timeout: Option<u64>,
    ) -> Result<Output> {
        let cwd = cwd
            .filter(|value| !value.is_empty())
            .unwrap_or(&self.config.cwd);
        let mut command = Command::new(&self.config.executable);
        command.arg("exec").arg("-w").arg(cwd);
        for key in &self.config.forward_env {
            if let Ok(value) = std::env::var(key) {
                command.arg("-e").arg(format!("{key}={value}"));
            }
        }
        for (key, value) in &self.config.env {
            if let Some(value) = value.as_str() {
                command.arg("-e").arg(format!("{key}={value}"));
            }
        }
        command.arg(&self.container_id);
        command.args(&self.config.interpreter);
        command.arg(&action.command);

        let output = self
            .execute_command(command, timeout.unwrap_or(self.config.timeout))
            .await?;
        self.check_finished(&output)?;
        Ok(output)
    }

    fn get_template_vars(&self) -> Value {
        merge_environment_template_vars(config_to_value(&self.config), Value::Object(Map::new()))
    }

    fn serialize(&self) -> Value {
        serde_json::json!({
            "info": {
                "config": {
                    "environment": config_to_value(&self.config),
                    "environment_type": "mini_swe_agent.environments.docker.DockerEnvironment",
                }
            }
        })
    }

    fn cleanup(&self) -> anyhow::Result<()> {
        let _ = std::process::Command::new(&self.config.executable)
            .arg("stop")
            .arg(&self.container_id)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        let _ = std::process::Command::new(&self.config.executable)
            .arg("rm")
            .arg("-f")
            .arg(&self.container_id)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        Ok(())
    }
}

impl Drop for DockerEnvironment {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

async fn run_captured(command: &mut Command, timeout: u64) -> anyhow::Result<(String, i32)> {
    command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let mut child = command
        .spawn()
        .context("could not spawn docker executable")?;
    let mut stdout = child.stdout.take().context("could not capture stdout")?;
    let mut stderr = child.stderr.take().context("could not capture stderr")?;
    let wait = async {
        let mut out = Vec::new();
        let mut err = Vec::new();
        let (status, out_result, err_result) = tokio::join!(
            child.wait(),
            stdout.read_to_end(&mut out),
            stderr.read_to_end(&mut err)
        );
        (status, out_result, err_result, out, err)
    };
    let (status, out_result, err_result, mut out, err) =
        tokio::time::timeout(Duration::from_secs(timeout.max(1)), wait)
            .await
            .with_context(|| format!("docker command timed out after {}s", timeout.max(1)))?;
    out_result.context("could not read stdout")?;
    err_result.context("could not read stderr")?;
    out.extend(err);
    Ok((
        normalize_newlines(String::from_utf8_lossy(&out).into_owned()),
        status.context("docker wait failed")?.code().unwrap_or(-1),
    ))
}
