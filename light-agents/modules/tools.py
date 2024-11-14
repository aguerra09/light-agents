import json
from typing import List
from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall
from utils.types import Tool


def get_tool_calls(response: ChatCompletion) -> List[ChatCompletionMessageToolCall]:
    return response.choices[0].message.tool_calls


def get_tool_attributes(tool_calls: ChatCompletionMessageToolCall):
    tool_call_id = tool_calls.id
    tool_function_name = tool_calls.function.name
    tool_arguments = json.loads(tool_calls.function.arguments)

    return tool_call_id, tool_function_name, tool_arguments


def call_function_tool(tools: List[Tool], tool_function_name: str, tool_arguments: dict):
    for tool in tools:
        if tool.name == tool_function_name:
            return tool.py_func(**tool_arguments)

