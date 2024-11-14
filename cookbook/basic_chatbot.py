from openai import OpenAI

from agent.light_agent import AgentConfig, LightAgent
from modules.conversation import Conversation

if __name__ == '__main__':
    client = OpenAI()
    config = AgentConfig(system_prompt="You are LightAI, a helpful assistant that speaks Spanish.")

    conversation = Conversation()

    agent = LightAgent(client=client, config=config, conversation=conversation)

    while True:
        user_input = input("Send a message to LightAI: > ")
        if user_input == "exit":
            break

        agent.execute([conversation.create_message(role='user', content=user_input)])
        print(response)