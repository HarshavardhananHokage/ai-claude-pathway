import json
import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

response = openai_client.embeddings.create(
    input="The weather is nice today.",
    model="text-embedding-3-small"
)

#print(json.dumps(response.model_dump(), indent=2))
print(len(response.data[0].embedding))