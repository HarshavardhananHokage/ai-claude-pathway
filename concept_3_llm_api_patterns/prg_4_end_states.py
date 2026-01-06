import json
import os

import openai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def stream_gpt_output(question, model="gpt-4o-mini"):
    client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

    stream = client.responses.create(
        input = question,
        model=model,
        max_output_tokens=30,
        #temperature=0.25,
        stream=True
    )

    for event in stream:
        if hasattr(event, "response"):
            print(f"Current Status of event type {event.type}: {event.response.status}")

def status_completed():
    client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

    response = client.responses.create(
        model="gpt-4o-mini",
        input="Explain HTTP 429 errors in 2 sentences"
    )

    print(json.dumps(response.model_dump(), indent=2))
    print(f"Status: {response.status}")

def retrieve_existing_response():
    client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))
    response = client.responses.retrieve(
        response_id="XXX"
    )
    print(json.dumps(response.model_dump(), indent=2))

def status_incomplete():
    client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input="Write a detailed 500-word review of a movie, but do not use the letter 'e' in any part of the text.",
            max_output_tokens=20
        )
        print(json.dumps(response.model_dump(), indent=2))
        print(f"Status: {response.status}")
    except openai.BadRequestError as e:
        print(json.dumps(e))

def status_invalid_model():
    client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

    try:
        response = client.responses.create(
            model="random-bullshit-go",
            input="Write a detailed 500-word review of a movie, but do not use the letter 'e' in any part of the text.",
        )
        print(json.dumps(response.model_dump(), indent=2))
        print(f"Status: {response.status}")
    except openai.BadRequestError as e:
        print(f"Received error: {e.status_code} Reason: {e.response.reason_phrase}")
        print(json.dumps(e.response.json()))

if __name__ == "__main__":
    # status_completed()
    # retrieve_existing_response()
    # status_incomplete()
    status_invalid_model()
    # stream_gpt_output("Give me poem like 'The Road Not Taken' about inspiration to achieve great things")