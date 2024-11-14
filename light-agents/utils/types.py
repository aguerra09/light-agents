from enum import Enum
from typing import Optional, List, Callable
from pydantic import BaseModel, Field


class Role(Enum):
    SYSTEM = 'system'
    ASSISTANT = 'assistant'
    USER = 'user'
    TOOL = 'tool'


class Tool(BaseModel):
    name: str
    py_func: Callable


class AgentConfig(BaseModel):
    model: str = Field(default="gpt-4o", description="OpenAI chat model (e.g. 'gpt-4', 'gpt-4o')")
    temperature: float = Field(default=0.0, description="Temperature for the model")
    system_prompt: str = Optional[Field(description="Agent system prompt")]
    tools_def: List[dict] = Optional[Field(description="Agent tools definition in OpenAI format")]
    tools_fn: List[Tool] = Optional[Field(description="Python functions to be used by the agent")]