from enum import Enum
from typing import Optional
from utils.types import Role

ROLES = Enum('system', 'assistant', 'user', 'tool')


class Conversation:
    """
    Base class to manage the Agent conversation.

    This class provides methods to manage the agent conversation, storing the message history generated for the agent.

    Attributes:
        history (list): List of messages sent by the user and generated for the agent.
    """
    def __init__(self):
        self.history = []

    @staticmethod
    def create_message(role: ROLES, content: str, tool_call_id: Optional[str] = None, tool_function_name: Optional[str] = None) -> dict:
        if role == Role.TOOL.value:
            return {"role": "tool", "tool_call_id": tool_call_id, "name": tool_function_name, "content": content}
        else:
            return {"role": role, "content": content}

    def get_history(self) -> list:
        return self.history

    def add_message(self, message: list[dict]):
        self.history = self.history + message

    def add_system_message(self, system_message: str) -> None:
        self.history.append(self.create_message(role=Role.SYSTEM.value, content=system_message))

    def add_user_message(self, user_message: str) -> None:
        self.history.append(self.create_message(role=Role.USER.value, content=user_message))

    def add_assistant_message(self, assistant_message: str) -> None:
        self.history.append(self.create_message(role=Role.ASSISTANT.value, content=assistant_message))

    def add_tool_message(self, tool_call_id: str, tool_function_name: str, results: str) -> None:
        self.history.append(
            self.create_message(
                role=Role.TOOL.value, tool_call_id=tool_call_id, tool_function_name=tool_function_name, content=results
            )
        )
