import json
import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def stream_gpt_output(question, model="gpt-4o-mini"):
    client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

    stream = client.responses.create(
        input = question,
        model=model,
        max_output_tokens=500,
        #temperature=0.25,
        stream=True
    )

    for event in stream:
        # print(json.dumps(event.model_dump(), indent=16))
        # print("\n\n")
        if event.type == "response.output_text.delta":
            print(event.delta, end="")
        elif event.type == "response.incomplete":
            print(f"\n\n{event.response.incomplete_details.reason}")

if __name__ == "__main__":
    stream_gpt_output("Give me poem like 'The Road Not Taken' about inspiration to achieve great things")