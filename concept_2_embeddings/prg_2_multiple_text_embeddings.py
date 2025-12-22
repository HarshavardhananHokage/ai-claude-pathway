import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

texts = [
    "The weather is pleasant today.",
    "It’s a beautiful day outside.",
    "The weather feels really good.",
    "It’s nice and comfortable outdoors.",
    "Today’s weather is quite enjoyable."
]

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

print(len(response.data))