import json
import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def stream_gpt_output(question, model="gpt-4o-mini"):
    client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"), timeout=1)

    stream = client.responses.create(
        input = question,
        model=model,
        max_output_tokens=1000,
        #temperature=0.25,
        stream=True
    )
    buffer = []
    try:
        for event in stream:
            if event.type == "response.output_text.delta":
                buffer.append(event.delta)
                print(event.delta, end="")
    except Exception as e:
        print("\n\nSTREAM FAILED")
        print("PARTIAL OUTPUT:")
        print("".join(buffer))
        print("ERROR:", type(e).__name__, e)

if __name__ == "__main__":
    stream_gpt_output("Give me poem like 'The Road Not Taken' about inspiration to achieve great things")