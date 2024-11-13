from typing import Optional
from pydantic import BaseModel, Field
from openai.types.chat import ChatCompletion

from openai import OpenAI

from utils.conversation import Conversation


class AgentConfig(BaseModel):
    model: str = Field(default="gpt-4o", description="OpenAI chat model (e.g. 'gpt-4', 'gpt-4o')")
    temperature: float = Field(default=0.0, description="Temperature for the model")
    system_prompt: str = Optional[Field(default=None, description="Agent system prompt")]


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
    """

    def __init__(self, client: OpenAI, config: AgentConfig, conversation: Conversation):
        self.client = client
        self.model = config.model
        self.temperature = config.temperature
        self.conversation = conversation
        if config.system_prompt:
            self.conversation.history = conversation.add_system_message(config.system_prompt)

    def get_chat_completions_response(self, messages: list) -> ChatCompletion:
        return self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
        )

    def execute(self, message: list):
        history = self.conversation.get_history()
        response = self.get_chat_completions_response(messages=history + message)
        self.conversation.add_assistant_message(response.choices[0].message.content)
        return response.choices[0].message.content
