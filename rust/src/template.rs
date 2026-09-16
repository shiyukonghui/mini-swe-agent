//! Thin Jinja-compatible template rendering layer.
//!
//! The reference project uses Jinja for user-facing prompts.  `minijinja`
//! gives us the same template shape (including slicing and `length`).  We also
//! provide a `tojson` filter that matches Python's default
//! `json.dumps(..., ensure_ascii=True)` behavior so observation prompts are
//! byte-for-byte stable across the Python and Rust implementations.

use serde_json::Value;

use crate::{AgentError, Result};

pub fn render(template: &str, context: &Value) -> Result<String> {
    let mut env = minijinja::Environment::new();
    env.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
    env.add_filter(
        "tojson",
        |value: minijinja::Value| -> std::result::Result<String, minijinja::Error> {
            let json = serde_json::to_string(&value).map_err(|error| {
                minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, error.to_string())
            })?;
            Ok(escape_non_ascii(json))
        },
    );
    let context = minijinja::Value::from_serialize(context);
    env.render_str(template, context).map_err(AgentError::other)
}

pub fn render_with_output(template: &str, context: &Value, output: &Value) -> Result<String> {
    let mut merged = context.as_object().cloned().unwrap_or_default();
    merged.insert("output".to_string(), output.clone());
    render(template, &Value::Object(merged))
}

fn escape_non_ascii(value: String) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        if character.is_ascii() {
            escaped.push(character);
            continue;
        }
        let code = character as u32;
        if code <= 0xFFFF {
            escaped.push_str(&format!("\\u{code:04x}"));
        } else {
            let code = code - 0x10000;
            let high = 0xD800 + (code >> 10);
            let low = 0xDC00 + (code & 0x3FF);
            escaped.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
        }
    }
    escaped
}
