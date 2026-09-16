//! Local subprocess environment.

use std::process::Stdio;
use std::time::Duration;

use anyhow::Context;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::io::AsyncReadExt;
use tokio::process::Command;

use crate::{Action, Environment, FlowInterrupt, Output, Result};

use super::{config_to_value, merge_environment_template_vars, normalize_newlines};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct LocalEnvironmentConfig {
    pub cwd: String,
    pub env: Map<String, Value>,
    pub timeout: u64,
}

impl Default for LocalEnvironmentConfig {
    fn default() -> Self {
        Self {
            cwd: String::new(),
            env: Map::new(),
            timeout: 30,
        }
    }
}

/// Executes each action in a fresh shell, mirroring `subprocess.run`.
pub struct LocalEnvironment {
    pub config: LocalEnvironmentConfig,
}

impl LocalEnvironment {
    pub fn new(config: LocalEnvironmentConfig) -> Self {
        Self { config }
    }

    pub fn from_value(value: Value) -> anyhow::Result<Self> {
        Ok(Self::new(
            serde_json::from_value(value).context("invalid local environment config")?,
        ))
    }

    async fn run_shell(
        &self,
        command: &str,
        cwd: &str,
        timeout: u64,
    ) -> anyhow::Result<(String, i32)> {
        #[cfg(unix)]
        let mut process = {
            let mut process = Command::new("sh");
            process.arg("-c").arg(command);
            use std::os::unix::process::CommandExt;
            process.process_group(0);
            process
        };
        #[cfg(windows)]
        let mut process = {
            let mut process = Command::new("cmd");
            process.arg("/C");
            // `cmd.exe` performs its own quote parsing. Passing the command
            // through a normal argument makes Rust quote it a second time and
            // mangles commands such as `python -c "print(42)"`. `raw_arg`
            // matches Python's `subprocess(..., shell=True)` behavior.
            process.raw_arg(command);
            process
        };

        process
            .current_dir(cwd)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        for (key, value) in &self.config.env {
            if let Some(value) = value.as_str() {
                process.env(key, value);
            }
        }

        let mut child = process.spawn().context("could not spawn shell")?;
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
                .with_context(|| format!("command timed out after {}s", timeout.max(1)))?;
        out_result.context("could not read stdout")?;
        err_result.context("could not read stderr")?;
        out.extend(err);
        let returncode = status.context("shell wait failed")?.code().unwrap_or(-1);
        Ok((
            normalize_newlines(String::from_utf8_lossy(&out).into_owned()),
            returncode,
        ))
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
impl Environment for LocalEnvironment {
    async fn execute(
        &self,
        action: &Action,
        cwd: Option<&str>,
        timeout: Option<u64>,
    ) -> Result<Output> {
        let cwd = cwd
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .or_else(|| (!self.config.cwd.is_empty()).then(|| self.config.cwd.clone()))
            .unwrap_or_else(|| {
                std::env::current_dir()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned()
            });
        let timeout = timeout.unwrap_or(self.config.timeout);
        let output = match self.run_shell(&action.command, &cwd, timeout).await {
            Ok((output, returncode)) => Output::success(output, returncode),
            Err(error) => Output::failure(error, "RuntimeError", ""),
        };
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
                    "environment_type": "mini_swe_agent.environments.local.LocalEnvironment",
                }
            }
        })
    }
}
