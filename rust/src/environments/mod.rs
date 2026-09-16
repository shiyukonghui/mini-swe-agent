//! Execution environments.

pub mod docker;
pub mod local;

pub use docker::{DockerEnvironment, DockerEnvironmentConfig};
pub use local::{LocalEnvironment, LocalEnvironmentConfig};

use std::sync::OnceLock;

use serde_json::{Map, Value};

#[derive(Clone, Debug)]
struct PlatformInfo {
    system: String,
    node: String,
    release: String,
    version: String,
    machine: String,
    processor: String,
}

impl PlatformInfo {
    fn detect() -> Self {
        let mut info = Self {
            system: platform_system(),
            node: std::env::var("COMPUTERNAME")
                .or_else(|_| std::env::var("HOSTNAME"))
                .unwrap_or_default(),
            release: String::new(),
            version: String::new(),
            machine: machine_name(std::env::consts::ARCH),
            processor: std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_default(),
        };

        #[cfg(windows)]
        {
            if let Ok(output) = std::process::Command::new("cmd")
                .arg("/C")
                .arg("ver")
                .output()
            {
                let text = String::from_utf8_lossy(&output.stdout);
                if let Some(start) = text.find("[Version ") {
                    if let Some(end) = text[start + 9..].find(']') {
                        let raw = &text[start + 9..start + 9 + end];
                        let mut parts = raw.split('.');
                        info.release = parts.next().unwrap_or_default().to_string();
                        info.version = parts.take(2).fold(info.release.clone(), |mut acc, part| {
                            acc.push('.');
                            acc.push_str(part);
                            acc
                        });
                    }
                }
            }
        }

        #[cfg(unix)]
        {
            if let Some(release) = uname("-r") {
                info.release = release;
            }
            if let Some(version) = uname("-v") {
                info.version = version;
            }
            if let Some(machine) = uname("-m") {
                info.machine = machine;
            }
            if let Some(node) = uname("-n") {
                info.node = node;
            }
            if let Some(processor) = uname("-p") {
                if !processor.is_empty() {
                    info.processor = processor;
                }
            }
        }

        info
    }
}

fn platform_info() -> &'static PlatformInfo {
    static INFO: OnceLock<PlatformInfo> = OnceLock::new();
    INFO.get_or_init(PlatformInfo::detect)
}

/// OS/process metadata exposed to prompt templates, mirroring the fields from
/// Python's `platform.uname()._asdict()` that the reference templates use.
pub fn platform_template_vars(extra: impl IntoIterator<Item = (String, Value)>) -> Value {
    let info = platform_info();
    let mut object = Map::new();
    object.insert("system".to_string(), Value::String(info.system.clone()));
    object.insert("release".to_string(), Value::String(info.release.clone()));
    object.insert("version".to_string(), Value::String(info.version.clone()));
    object.insert("machine".to_string(), Value::String(info.machine.clone()));
    object.insert("node".to_string(), Value::String(info.node.clone()));
    object.insert(
        "processor".to_string(),
        Value::String(info.processor.clone()),
    );
    for (key, value) in std::env::vars() {
        object.insert(key, Value::String(value));
    }
    for (key, value) in extra {
        object.insert(key, value);
    }
    Value::Object(object)
}

fn platform_system() -> String {
    match std::env::consts::OS {
        "macos" => "Darwin".to_string(),
        "windows" => "Windows".to_string(),
        "linux" => "Linux".to_string(),
        other => {
            let mut chars = other.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        }
    }
}

fn machine_name(arch: &str) -> String {
    match arch {
        "x86_64" => "AMD64".to_string(),
        "x86" | "i686" => "x86".to_string(),
        "aarch64" => "ARM64".to_string(),
        other => other.to_string(),
    }
}

#[cfg(unix)]
fn uname(flag: &str) -> Option<String> {
    let output = std::process::Command::new("uname")
        .arg(flag)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let value = text.trim().to_string();
    (!value.is_empty()).then_some(value)
}

pub(crate) fn config_to_value<T: serde::Serialize>(config: &T) -> Value {
    serde_json::to_value(config).unwrap_or(Value::Null)
}

pub(crate) fn merge_environment_template_vars(config: Value, kwargs: Value) -> Value {
    [config, platform_template_vars(Vec::new()), kwargs]
        .into_iter()
        .fold(Value::Object(Map::new()), |base, value| {
            crate::json_merge(base, value)
        })
}

/// Match Python `subprocess.run(text=True)` universal-newline behavior.
pub(crate) fn normalize_newlines(value: String) -> String {
    value.replace("\r\n", "\n").replace('\r', "\n")
}
