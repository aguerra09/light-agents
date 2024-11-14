from typing import List
from openai.types.chat import ChatCompletion

from openai import OpenAI

from modules.conversation import Conversation
from modules.tools import get_tool_calls, call_function_tool, get_tool_attributes
from utils.types import AgentConfig


class LightAgent:
    """
    Base class to build OpenAI Chat Completions based agents.

    This class provides the core functionality to interact with Chat Completions OpenAI LLM models,
    managing the conversation, obtaining responses from the model and executing tools.

    Attributes:
        client (OpenAI): OpenAI client.
        model (str): OpenAI chat model (e.g. 'gpt-4', 'gpt-4o').
        temperature (float): temperature of the model.
        conversation (Conversation): class to manage and store the conversation history.
        tools_def (List[dict]): list of tool definitions in OpenAI format.
        tools_fn (List[Callables]): list of callable functions available for the agent.
    """

    def __init__(self, client: OpenAI, config: AgentConfig, conversation: Conversation):
        self.client = client
        self.model = config.model
        self.temperature = config.temperature
        self.conversation = conversation
        self.tools_def = config.tools_def
        self.tools_fn = config.tools_fn
        if config.system_prompt:
            self.conversation.add_system_message(config.system_prompt)

    def get_chat_completions_response(self, messages: list) -> ChatCompletion:
        return self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
            tools=self.tools_def,
        )

    def execute(self, message: List[dict[str, str]]) -> None:
        self.conversation.add_message(message)
        response = self.get_chat_completions_response(messages=self.conversation.get_history())

        tool_calls = get_tool_calls(response)
        if tool_calls is not None:
            call_id, tool_name, tool_args = get_tool_attributes(tool_calls[0])
            result = call_function_tool(self.tools_fn, tool_name, tool_args)
            self.conversation.add_tool_message(call_id, tool_name, result)
        else:
            self.conversation.add_assistant_message(response.choices[0].message.content)
