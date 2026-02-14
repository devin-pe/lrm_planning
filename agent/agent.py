# -*- coding: utf-8 -*-
"""
Code Executing Agent with E2B Sandbox.

A flexible code executing agent that "lives" in an E2B sandbox environment.
Unlike tool-based agents with rigid predefined actions, this agent has full
terminal flexibility - it can write scripts, create files, run code, and
execute shell commands dynamically to solve complex problems.

Usage:
LangGraph Studio (interactive chat + tool visibility)
    - code_agent is registered in langgraph.json
    - Run: langgraph dev
    - Open Studio, select "code_agent" graph, chat and inspect tools.
"""
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from typing import List, Optional
from e2b_interpreter import get_sandbox
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import Annotated, TypedDict
from prompts import CODE_SYSTEM_PROMPT

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault('LANGCHAIN_PROJECT', 'code-agent')


class InputState(TypedDict):
    """Minimal input schema for LangGraph Studio."""
    messages: Annotated[List[AnyMessage], add_messages]


class State(TypedDict):
    """Full state schema for the code executing agent."""
    messages: Annotated[List[AnyMessage], add_messages]
    sandbox_id: Optional[str]  # Track sandbox session


@tool
def execute_python(code: str) -> str:
    """
    Execute Python code in the sandbox environment.

    The sandbox has:
    - Full Python environment with pandas, numpy, matplotlib, etc.
    - Data files pre-loaded in /home/user/data/
    - Ability to create and import custom modules
    - Full filesystem access within the sandbox

    Use this for:
    - Data analysis and transformations
    - Running Python scripts
    - Complex computations
    - Any Python code execution

    Args:
        code: Python code to execute. Can be multi-line.

    Returns:
        Execution results including output, logs, and any errors.
    """
    sbx = get_sandbox()
    execution = sbx.run_code(code)

    if execution.error:
        return f'Error: {execution.error.name}: {execution.error.value}'

    output_parts = []
    if execution.text:
        output_parts.append(f'Output:\n{execution.text}')
    if execution.logs.stdout:
        output_parts.append(f'Stdout:\n{execution.logs.stdout}')
    if execution.logs.stderr:
        output_parts.append(f'Stderr:\n{execution.logs.stderr}')

    return '\n\n'.join(output_parts) if output_parts else '(executed successfully, no output)'


@tool
def run_shell(command: str) -> str:
    """
    Execute a shell command in the sandbox environment.

    Full bash access within the sandbox for:
    - File operations: ls, cat, head, tail, mkdir, rm, mv, cp
    - Text processing: grep, sed, awk, sort, uniq, wc
    - Package management: pip install, pip list
    - Running scripts: python script.py, bash script.sh
    - System commands: pwd, whoami, env, which

    Args:
        command: Shell command to execute.

    Returns:
        Command output (stdout and stderr).
    """
    sbx = get_sandbox()
    result = sbx.commands.run(command)

    output_parts = []
    if result.stdout:
        output_parts.append(result.stdout)
    if result.stderr:
        output_parts.append(f'STDERR:\n{result.stderr}')
    if result.exit_code != 0:
        output_parts.append(f'Exit code: {result.exit_code}')

    return '\n'.join(output_parts) if output_parts else '(no output)'


@tool
def write_file(path: str, content: str) -> str:
    """
    Write content to a file in the sandbox.

    Use this to:
    - Create Python scripts to run later
    - Write configuration files
    - Save analysis results
    - Create data files

    The path should be within the sandbox filesystem.
    Common locations:
    - /home/user/scripts/ - for custom scripts
    - /home/user/data/ - for data files
    - /home/user/output/ - for results

    Args:
        path: File path in the sandbox (e.g., '/home/user/scripts/analyze.py')
        content: File content to write.

    Returns:
        Confirmation message.
    """
    sbx = get_sandbox()
    sbx.files.write(path, content)
    return f'File written successfully: {path}'


@tool
def read_file(path: str) -> str:
    """
    Read content from a file in the sandbox.

    Use this to:
    - Read data files
    - Check script contents
    - View analysis results
    - Inspect configuration

    Args:
        path: File path in the sandbox to read.

    Returns:
        File content or error message.
    """
    sbx = get_sandbox()
    try:
        content = sbx.files.read(path)
        return content
    except Exception as e:
        return f'Error reading file: {str(e)}'


# List of tools available to the agent
tools = [execute_python, run_shell, write_file, read_file]


model = ChatOpenAI(
    openai_api_key=os.getenv('OPENROUTER_API_KEY'),
    openai_api_base='https://openrouter.ai/api/v1',
    temperature=0.0,
    model='deepseek/deepseek-r1',
    timeout=1200,
    stream_usage=True,
).bind_tools(tools, parallel_tool_calls=False)


# Graph Nodes
def agent_node(state: State):
    """Core LLM invocation with tools bound."""
    messages = state['messages']
    sys_prompt = SystemMessage(content=CODE_SYSTEM_PROMPT)
    response = model.invoke([sys_prompt] + messages)
    return {'messages': [response]}


def tool_router(state: State):
    """Route to tools or end based on whether the LLM made tool calls."""
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return 'tools'
    return END


# Tool node that executes the tools
tool_node = ToolNode(tools)

# Graph Definition
graph_builder = StateGraph(State, input_schema=InputState)

# Add nodes
graph_builder.add_node('agent', agent_node)
graph_builder.add_node('tools', tool_node)

# Add edges
graph_builder.add_edge(START, 'agent')
graph_builder.add_conditional_edges('agent', tool_router, ['tools', END])
graph_builder.add_edge('tools', 'agent')

# Compile the graph
graph = graph_builder.compile()
