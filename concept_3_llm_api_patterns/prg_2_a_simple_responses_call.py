import json
import os

from openai import OpenAI
from dotenv import  load_dotenv

load_dotenv()

def a_simple_openai_call():
    client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

    response = client.responses.create(
        model="gpt-4o-mini",
        input="Explain HTTP 429 errors in 2 sentences"
    )

    # print(json.dumps(response.model_dump(), indent=2))
    # print(json.dumps(response.output[0].content[0].model_dump(), indent=2))
    print(json.dumps(response.usage.model_dump(), indent=2))

if __name__ == "__main__":
    a_simple_openai_call()