import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("RATE_LIMIT_KEY"))

def a_simple_openai_call(i: int):
    return client.responses.create(
        model="gpt-4o-mini",
        input=f"Explain HTTP 429 errors in 2 sentences. Request {i}",
        max_output_tokens=200
    )

def trigger_429():
    with ThreadPoolExecutor(max_workers=25) as executor:
        futures = [
            executor.submit(a_simple_openai_call, i)
            for i in range(50)
        ]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                if getattr(e, "status_code", None) == 429:
                    resp = e.response  # httpx.Response

                    print("\n=== HTTP STATUS ===")
                    print(resp.status_code, resp.reason_phrase)

                    print("\n=== HEADERS ===")
                    print(json.dumps(dict(resp.headers), indent=2))

                    print("\n=== BODY (JSON) ===")
                    try:
                        print(json.dumps(resp.json(), indent=2))
                    except Exception:
                        print(resp.text)

if __name__ == "__main__":
    trigger_429()