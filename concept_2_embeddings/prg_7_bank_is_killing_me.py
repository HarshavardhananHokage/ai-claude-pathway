from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

texts = [
    # Keep structure identical, only change the adjective
    "The bank was steep.",          # Riverbank (steep = geographic feature)
    "The bank was closed.",         # Ambiguous
    "The bank was unstable.",       # Could be either (financial instability OR riverbank erosion)
    "The bank was empty.",          # Likely financial (empty of money/people)
    "The bank was muddy.",          # Clearly riverbank (physical description)
    "The bank was profitable.",     # Clearly financial (business metric)
]

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

embeddings = [np.array(item.embedding) for item in response.data]

print("SAME STRUCTURE, DIFFERENT ADJECTIVES:\n")

print("Riverbank context:")
print(f"'steep' vs 'muddy': {cosine_similarity([embeddings[0]], [embeddings[4]])[0][0]:.4f}")
print(f"'steep' vs 'unstable': {cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]:.4f}\n")

print("Financial context:")
print(f"'closed' vs 'profitable': {cosine_similarity([embeddings[1]], [embeddings[5]])[0][0]:.4f}")
print(f"'closed' vs 'empty': {cosine_similarity([embeddings[1]], [embeddings[3]])[0][0]:.4f}\n")

print("Cross-domain:")
print(f"'steep' (riverbank) vs 'profitable' (financial): {cosine_similarity([embeddings[0]], [embeddings[5]])[0][0]:.4f}")
print(f"'muddy' (riverbank) vs 'empty' (financial): {cosine_similarity([embeddings[4]], [embeddings[3]])[0][0]:.4f}")