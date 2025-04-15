# Guidelines for using Python for MCP Tools

The following guidelines apply to any method or function marked with the @mcp.tool() decorator.

- Parameters should not have any of the following types: `Any`, `object`, `Optional[T]`, `Union[T]`.
- For a given parameter `x` of type `T` that has a default value, do not use type `x : T | None = None`. Instead, use `x: T = None` and handle defaults within the method body itself.
- Always include method docstrings and make sure to given proper examples of valid inputs especially when no type hints are present.

When this rule is applied, please remember to explicitly mention it.


# Guidelines for String Formatting in CrewAI Tasks and Agent Descriptions

When writing task descriptions or agent backstories for CrewAI:

- Avoid using curly braces `{}` in string literals as they will be interpreted as format specifiers by CrewAI's template engine
- If you need to show JSON examples, use `dict()` notation instead of JSON literal syntax: 
  - Use `dict(key="value")` instead of `{"key": "value"}`
  - For nested structures, use `dict(outer=dict(inner="value"))` instead of `{"outer": {"inner": "value"}}`
- Only use named format variables that are explicitly provided in the `inputs` dictionary when calling `kickoff()`
- If you must include curly braces in your examples, escape them by doubling: `{{` and `}}` 
- When showing examples of command lists, use Python list syntax with `dict()` objects:
  ```python
  [dict(command="get_actors", params=dict()), dict(command="spawn_actor", params=dict(name="Cube"))]
  ```

Following these guidelines will prevent the common `IndexError: Replacement index out of range` and `KeyError: 'variable_name'` errors.


# Guidelines for Tool and Plugin Naming in CrewAI

When referencing tools or plugins in CrewAI task descriptions and agent backstories:

- **Always use the exact tool name**: Refer to tools by their actual function name, not by a description or translation
  - Correct: `'get_available_commands'` 
  - Incorrect: `'Listar Comandos MCP Disponibles'` or `'Available Commands Tool'`

- **Use single quotes around tool names**: This helps avoid parsing issues with JSON and string formatting
  - Correct: `'execute_mcp_command'`
  - Avoid: `"execute_mcp_command"`

- **Mention tool names consistently**: Use the same name throughout all descriptions and documentation
  - If the tool function is named `spawn_actor`, always refer to it as `'spawn_actor'`, not variants like `'create_actor'`

- **Test tool invocation**: When implementing new tools, always verify that the agent can successfully invoke them before deployment
  - In your development process, examine logs to confirm tools are being called with their correct names

These naming conventions help ensure that AI agents can correctly identify and call the tools available to them, preventing errors like `Action 'Tool Name' doesn't exist` that cause task failures.