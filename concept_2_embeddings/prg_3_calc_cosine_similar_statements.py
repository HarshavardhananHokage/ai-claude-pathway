import json
import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

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

embeddings = [item.embedding for item in response.data]

for i in range(0, len(embeddings)):
    for j in range(i+1, len(embeddings)):
        embeddings1 = np.array(embeddings[i])
        embeddings2 = np.array(embeddings[j])
        similarity = cosine_similarity([embeddings1], [embeddings2])[0][0]
        print(f"Similarity between statement {i+1} and {j+1} in list is {similarity}")

print(len(response.data))