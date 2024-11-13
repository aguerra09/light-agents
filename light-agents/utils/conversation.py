from enum import Enum


class Role(Enum):
    SYSTEM = 'system'
    ASSISTANT = 'assistant'
    USER = 'user'


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
    def create_message(role: str, content: str) :
        return {"role": role, "content": content}

    def get_history(self) -> list:
        return self.history

    def add_system_message(self, system_message: str) -> list:
        self.history.append(self.create_message(role=Role.SYSTEM.value, content=system_message))
        return self.history

    def add_user_message(self, user_message: str) -> list:
        self.history.append(self.create_message(role=Role.USER.value, content=user_message))
        return self.history

    def add_assistant_message(self, assistant_message: str) -> list:
        self.history.append(self.create_message(role=Role.ASSISTANT.value, content=assistant_message))
        return self.history
