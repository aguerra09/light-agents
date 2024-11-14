from openai import OpenAI

from agent.light_agent import LightAgent, AgentConfig
from modules.conversation import Conversation
from utils.types import Tool

SYSTEM_PROMPT = """
    **You are LightAI, an AI intelligent assistant. Your main goal is to call tools properly.**
    
    *Tools:*
    This are the tools available for you to use:
    * `read_file`- tool for reading files given a path or filename returning his content. Tool parameters:
        * `file_path` (str) - path of the file to read.
    
    *Instructions:*
    1) Determine if the user wants to read a file.
    2) If the user wants to read a file invoke the tool `read_file` with the `file_path` parameter provided for the user.
    3) If the user doesn't want to read a file ask him for any other help.
"""

tool_def = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Return the content of a file provided for the user. The function is called when the user needs to read a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path of the file to read",
                    },
                },
                "required": ["file_path"]
            }
        }
    }
]


def read_file(file_path):
    with open(file_path, "r") as f:
        return f.read()


if __name__ == '__main__':
    client = OpenAI()
    read_file_tool = Tool(name="read_file", py_func=read_file)
    config = AgentConfig(model="gpt-4o-mini", temperature=0, system_prompt=SYSTEM_PROMPT, tools_def=tool_def, tools_fn=[read_file_tool])
    conversation = Conversation()

    agent_with_tools = LightAgent(client=client, config=config, conversation=conversation)

    while True:
        user_input = input("Send a question to LightAI: > ")
        if user_input == "exit":
            break

        agent_with_tools.execute([conversation.create_message(role='user', content=user_input)])
        print(conversation.history[-1])
