from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

texts = [
    # Original ambiguous sentences
    "The bank was steep.",
    "The bank was closed.",

    # Add context to disambiguate "closed"
    "The bank was closed, so I couldn't deposit my paycheck.",  # Clearly financial
    "The bank was closed due to flooding, blocking river access.",  # Clearly riverbank

    # Add more riverbank examples
    "We walked along the steep riverbank.",
    "The hiking trail followed the bank of the river.",
]

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

embeddings = [np.array(item.embedding) for item in response.data]

print("ORIGINAL (ambiguous):")
print(f"'bank was steep' vs 'bank was closed': {cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]:.4f}\n")

print("DISAMBIGUATED:")
print(
    f"'bank was steep' vs 'bank closed + deposit paycheck' (financial): {cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]:.4f}")
print(
    f"'bank was steep' vs 'bank closed + flooding' (riverbank): {cosine_similarity([embeddings[0]], [embeddings[3]])[0][0]:.4f}\n")

print("EXPLICIT RIVERBANK:")
print(f"'bank was steep' vs 'steep riverbank': {cosine_similarity([embeddings[0]], [embeddings[4]])[0][0]:.4f}")
print(f"'bank was steep' vs 'bank of the river': {cosine_similarity([embeddings[0]], [embeddings[5]])[0][0]:.4f}")