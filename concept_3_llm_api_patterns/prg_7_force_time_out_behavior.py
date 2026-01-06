import json
import os
import pprint

import openai
from openai import OpenAI
from dotenv import  load_dotenv

load_dotenv()

def force_timeout_behavior():
    client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"), timeout=1)

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input="Explain HTTP 429 errors in 2 sentences"
        )
        print(json.dumps(response.usage.model_dump(), indent=2))
    except openai.APITimeoutError as e:
        print(f"Status: {e.code}")
        pprint.pprint(e)

    # response = client.responses.create(
    #         model="gpt-4o-mini",
    #         input="Explain HTTP 429 errors in 2 sentences"
    #     )

if __name__ == "__main__":
    force_timeout_behavior()