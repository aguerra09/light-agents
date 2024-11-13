from openai import OpenAI

from agent.light_agent import AgentConfig, LightAgent
from utils.conversation import Conversation

if __name__ == '__main__':
    client = OpenAI()
    config = AgentConfig(system_prompt="You are LightAI, a helpful assistant that speaks Spanish.")

    conversation = Conversation()

    agent = LightAgent(client=client, config=config, conversation=conversation)

    while True:
        user_input = input("Send a message to LightAI: > ")
        if user_input == "exit":
            break

        response = agent.execute(conversation.add_user_message(user_input))
        print(response)