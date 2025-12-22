import json
import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

texts = [
    "Distributed databases must balance consistency, availability, and partition tolerance under network failures.",
    "Photosynthesis converts sunlight into chemical energy using chlorophyll inside plant chloroplasts.",
    "Central banks raise interest rates to control inflation and stabilize long-term economic growth.",
    "The Roman Empire relied on an extensive road network to administer distant provinces efficiently.",
    "A batsman adjusts footwork and timing to counter fast swing bowling on a damp cricket pitch."
]

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

embeddings = [item.embedding for item in response.data]

for i in range(0, len(embeddings)):
    for j in range(i+1, len(embeddings)):
        embeddings1 = np.array(embeddings[i])
        embeddings2 = np.array(embeddings[j])
        similarity = cosine_similarity([embeddings1], [embeddings2])[0][0]
        print(f"Similarity between statement {i+1} and {j+1} in list is {similarity}")

print(len(response.data))