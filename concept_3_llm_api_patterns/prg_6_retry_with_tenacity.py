from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception_type, wait_exponential
)
from openai import OpenAI, BadRequestError
import os
import pprint
from dotenv import load_dotenv
import time
from datetime import datetime

load_dotenv()

def on_before_sleep(retry_state):
    wall = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    mono = time.monotonic()

    last = getattr(on_before_sleep, "last", None)
    delta = 0 if last is None else mono - last

    print(f"[{wall}] Δ={delta:.3f}s")
    on_before_sleep.last = mono

def on_retries_exhausted(retry_state):
    print("=== Retries exhausted ===")
    pprint.pprint(retry_state)
    # re-raise the last exception
    return retry_state.outcome.result()

@retry(
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(BadRequestError),
    retry_error_callback=on_retries_exhausted,
)
def basic_invalid_model_openai_call():
    client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))
    return client.responses.create(
        model="random-bullshit-go",
        input="test"
    )

@retry(
    wait=wait_exponential(multiplier=1, min=1, max=60),
    after=on_before_sleep,
    stop=stop_after_attempt(5)
)
def basic_invalid_model_openai_call_with_exp_backoff():
    client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))
    print("Attempting jitter")

    return client.responses.create(
        model="random-bullshit-go",
        input="test"
    )

if __name__ == "__main__":
    basic_invalid_model_openai_call_with_exp_backoff()