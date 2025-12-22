from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

texts = [
    # SHORT: Same meaning, high overlap
    "The weather is nice.",
    "The weather is pleasant.",

    # MEDIUM: Same meaning, moderate overlap
    "I enjoyed the sunny weather today.",
    "Today's pleasant climate was quite enjoyable.",

    # LONG: Same meaning, low overlap (heavy paraphrasing)
    "The company's quarterly revenue exceeded analyst expectations primarily due to strong consumer demand in the technology sector.",
    "Our firm's financial performance this quarter surpassed Wall Street projections because of robust customer purchasing activity in tech products.",

    # LONG: Different meaning, but some word overlap
    "The company's quarterly revenue declined significantly due to weak consumer demand in the technology sector.",
    "Our firm's financial performance this quarter surpassed Wall Street projections because of robust customer purchasing activity in tech products.",
]

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

embeddings = [np.array(item.embedding) for item in response.data]

print("LENGTH vs SIMILARITY SCORES:\n")

print("SHORT (same meaning, high word overlap):")
sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
print(f"'weather nice' vs 'weather pleasant': {sim:.4f}\n")

print("MEDIUM (same meaning, moderate paraphrasing):")
sim = cosine_similarity([embeddings[2]], [embeddings[3]])[0][0]
print(f"'enjoyed sunny weather' vs 'pleasant climate enjoyable': {sim:.4f}\n")

print("LONG (same meaning, heavy paraphrasing):")
sim = cosine_similarity([embeddings[4]], [embeddings[5]])[0][0]
print(f"'revenue exceeded expectations' vs 'performance surpassed projections': {sim:.4f}\n")

print("LONG (OPPOSITE meaning, but similar words):")
sim = cosine_similarity([embeddings[6]], [embeddings[7]])[0][0]
print(f"'revenue DECLINED' vs 'performance SURPASSED': {sim:.4f}")
print("(Both talk about quarterly performance, but opposite results)")