//! Configuration loading and recursive merging.
//!
//! Built-in reference configs are embedded so a single binary remains
//! self-contained on machines where the Python source tree is not present.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use serde_json::{Map, Value};

use crate::json_merge;

const DEFAULT_CONFIG: &str = include_str!("../../src/minisweagent/config/default.yaml");
const MINI_CONFIG: &str = include_str!("../../src/minisweagent/config/mini.yaml");
const MINI_TEXT_BASED_CONFIG: &str =
    include_str!("../../src/minisweagent/config/mini_textbased.yaml");

/// Name -> YAML source for built-in configs.
pub fn builtin_config(name: &str) -> Option<&'static str> {
    let stem = Path::new(name)
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or(name);
    match stem {
        "default" => Some(DEFAULT_CONFIG),
        "mini" => Some(MINI_CONFIG),
        "mini_textbased" => Some(MINI_TEXT_BASED_CONFIG),
        _ => None,
    }
}

/// Directory for the global mini-SWE-agent config and default trajectory.
pub fn global_config_dir() -> PathBuf {
    std::env::var_os("MSWEA_GLOBAL_CONFIG_DIR")
        .map(PathBuf::from)
        .or_else(|| dirs::config_dir().map(|path| path.join("mini-swe-agent")))
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn global_config_file() -> PathBuf {
    global_config_dir().join(".env")
}

/// Load the global `.env` file if it exists.  This is intentionally
/// best-effort: a missing global config is a normal first-run state.
pub fn load_global_config() {
    let dir = global_config_dir();
    let _ = std::fs::create_dir_all(&dir);
    let env_file = dir.join(".env");
    if env_file.exists() {
        let _ = dotenvy::from_path(&env_file);
    }
}

fn parse_yaml(source: &str) -> anyhow::Result<Value> {
    let yaml: serde_yaml::Value =
        serde_yaml::from_str(source).context("invalid YAML configuration")?;
    serde_json::to_value(yaml).context("could not convert YAML configuration to JSON value")
}

fn candidate_paths(spec: &str) -> Vec<PathBuf> {
    let raw = PathBuf::from(spec);
    let with_suffix = if raw.extension().and_then(|value| value.to_str()) == Some("yaml")
        || raw.extension().and_then(|value| value.to_str()) == Some("yml")
    {
        raw.clone()
    } else {
        raw.with_extension("yaml")
    };
    let mut candidates = vec![raw.clone(), with_suffix];
    if let Some(dir) = std::env::var_os("MSWEA_CONFIG_DIR") {
        let dir = PathBuf::from(dir);
        candidates.push(dir.join(&raw));
        candidates.push(dir.join(raw.with_extension("yaml")));
    }
    if let Some(manifest_dir) = option_env!("CARGO_MANIFEST_DIR") {
        let builtin = PathBuf::from(manifest_dir)
            .join("..")
            .join("src")
            .join("minisweagent")
            .join("config");
        candidates.push(builtin.join(&raw));
        candidates.push(builtin.join(raw.with_extension("yaml")));
    }
    candidates
}

/// Return the first existing filesystem config path for `spec`.
pub fn get_config_path(spec: &str) -> Option<PathBuf> {
    candidate_paths(spec)
        .into_iter()
        .find(|candidate| candidate.exists())
}

/// Interpret a CLI key-value spec such as `model.model_name=openai/gpt-4o`.
pub fn key_value_spec_to_nested_dict(spec: &str) -> anyhow::Result<Value> {
    let (key, value) = spec
        .split_once('=')
        .with_context(|| format!("invalid config spec `{spec}`: expected key=value"))?;
    let value = serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_string()));
    let keys: Vec<&str> = key.split('.').collect();
    if keys.iter().any(|key| key.is_empty()) {
        bail!("invalid config spec `{spec}`: empty config key");
    }
    let mut root = Map::new();
    let mut current = &mut root;
    for key in &keys[..keys.len() - 1] {
        current = current
            .entry((*key).to_string())
            .or_insert_with(|| Value::Object(Map::new()))
            .as_object_mut()
            .with_context(|| {
                format!("invalid config spec `{spec}`: key `{key}` is not an object")
            })?;
    }
    current.insert(keys[keys.len() - 1].to_string(), value);
    Ok(Value::Object(root))
}

/// Load one configuration spec.  Specs may be key-value pairs, built-in
/// names, or filesystem paths.
pub fn get_config_from_spec(spec: &str) -> anyhow::Result<Value> {
    if spec.contains('=') {
        return key_value_spec_to_nested_dict(spec);
    }
    if let Some(source) = builtin_config(spec) {
        return parse_yaml(source);
    }
    let path =
        get_config_path(spec).with_context(|| format!("could not find config file for {spec}"))?;
    let source = std::fs::read_to_string(&path)
        .with_context(|| format!("could not read config file {}", path.display()))?;
    parse_yaml(&source)
}

/// Recursively merge JSON object values.  Later values win; `null` values are
/// treated as "unset" and therefore skipped, matching the Python sentinel.
pub fn recursive_merge(values: impl IntoIterator<Item = Value>) -> Value {
    values
        .into_iter()
        .fold(Value::Object(Map::new()), |base, patch| {
            json_merge(base, patch)
        })
}
