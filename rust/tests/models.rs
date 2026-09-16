use serde_json::json;

use mini_swe_agent::models::{parse_regex_actions, parse_toolcall_actions};
use mini_swe_agent::AgentError;

#[test]
fn parses_valid_bash_tool_call() {
    let actions = parse_toolcall_actions(
        &[json!({
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": "{\"command\":\"ls -la\"}"}
        })],
        "{{ error }}",
        Some("stop"),
    )
    .unwrap();
    assert_eq!(actions.len(), 1);
    assert_eq!(actions[0].command, "ls -la");
    assert_eq!(actions[0].tool_call_id.as_deref(), Some("call_1"));
}

#[test]
fn rejects_tool_call_without_command() {
    let error = parse_toolcall_actions(
        &[json!({
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": "{}"}
        })],
        "{{ error }}",
        Some("stop"),
    )
    .unwrap_err();
    assert!(matches!(error, AgentError::Format(_)));
}

#[test]
fn parses_exactly_one_text_action() {
    let actions = parse_regex_actions(
        "thinking\n```mswea_bash_command\npwd\n```\n",
        r"```mswea_bash_command\s*\n(.*?)\n```",
        "{{ error }}",
        None,
    )
    .unwrap();
    assert_eq!(actions[0].command, "pwd");

    let error = parse_regex_actions(
        "```mswea_bash_command\npwd\n```\n```mswea_bash_command\nls\n```",
        r"```mswea_bash_command\s*\n(.*?)\n```",
        "{{ error }}",
        None,
    )
    .unwrap_err();
    assert!(matches!(error, AgentError::Format(_)));
}

#[test]
fn renders_output_template_with_slicing_and_json_filter() {
    let template = "{% if output.output | length < 10 %}{{ output.output }}{% else %}{{ output.output[:5] }}{% endif %}";
    let output = json!({"output": "abcdefghijkl", "returncode": 0, "exception_info": ""});
    let rendered =
        mini_swe_agent::template::render_with_output(template, &json!({}), &output).unwrap();
    assert_eq!(rendered, "abcde");
}
