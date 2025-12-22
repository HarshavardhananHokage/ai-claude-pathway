import json
import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

texts = [
    # Weather / environment (similar cluster)
    "The weather is warm and sunny this afternoon.",
    "A cool breeze makes the evening weather pleasant.",
    "Heavy clouds suggest rain later in the day.",
    "The forecast predicts mild temperatures throughout the week.",

    # Technology / distributed systems (similar cluster)
    "Distributed systems must handle partial failures without losing availability.",
    "Event-driven architectures improve scalability by decoupling services.",
    "Consensus algorithms help nodes agree despite network partitions.",
    "Caching layers reduce latency in high-throughput applications.",

    # Finance / economics (similar cluster)
    "Central banks adjust interest rates to manage inflation.",
    "Stock markets react quickly to unexpected economic data.",
    "Long-term investors focus on fundamentals rather than short-term volatility.",
    "Rising energy prices contribute to broader inflationary pressure.",

    # Biology / science (similar cluster)
    "DNA replication ensures genetic information is passed to new cells.",
    "Enzymes accelerate chemical reactions inside living organisms.",
    "Photosynthesis allows plants to convert sunlight into usable energy.",

    # Sports (related but distinct)
    "The football team changed tactics at halftime to regain control.",
    "A tennis player relies on footwork and timing to return fast serves.",
    "Cricket bowlers exploit pitch conditions to generate swing and seam movement.",

    # History / culture (mostly unrelated to others)
    "Ancient trade routes connected distant civilizations through commerce.",
    "Medieval castles were designed to withstand prolonged sieges."
]

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

embeddings = [item.embedding for item in response.data]
answers = []

for i in range(0, len(embeddings)):
    for j in range(i+1, len(embeddings)):
        embeddings1 = np.array(embeddings[i])
        embeddings2 = np.array(embeddings[j])
        similarity = cosine_similarity([embeddings1], [embeddings2])[0][0]
        insert_tuple = (similarity, texts[i], texts[j])
        answers.append(insert_tuple)
        print(f"Similarity between statement {i+1} and {j+1} in list is {similarity}")

answers.sort(reverse=True)
print("The 3 most similar statements and their similarity score")
for item in answers[:3]:
    print(item)
print("\n")
print("The 3 most dissimilar statements and their similarity score")
for item in answers[-3:]:
    print(item)
print(len(response.data))