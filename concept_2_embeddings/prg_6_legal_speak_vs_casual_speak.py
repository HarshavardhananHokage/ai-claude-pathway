from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

texts = [
    # Casual user queries
    "Can my landlord kick me out?",
    "What happens if I don't pay rent?",
    "Do I have to let my landlord inspect my apartment?",

    # Formal legal documents
    "Grounds for eviction pursuant to landlord-tenant statutes include non-payment of rent and lease violations.",
    "A landlord may terminate tenancy upon documented failure to remit monthly rental payments as stipulated in the lease agreement.",
    "Property owners retain the right to conduct periodic inspections with proper notice as provided under residential tenancy regulations.",
]

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

embeddings = [np.array(item.embedding) for item in response.data]

# Test casual vs formal matches
print("Casual Query vs Formal Document Similarities:\n")

# Query 1 vs Doc 1 (both about eviction)
sim = cosine_similarity([embeddings[0]], [embeddings[3]])[0][0]
print(f"'Can landlord kick me out?' vs 'Grounds for eviction...'")
print(f"Similarity: {sim:.4f}\n")

# Query 2 vs Doc 2 (both about non-payment)
sim = cosine_similarity([embeddings[1]], [embeddings[4]])[0][0]
print(f"'What if I don't pay rent?' vs 'Failure to remit payments...'")
print(f"Similarity: {sim:.4f}\n")

# Query 3 vs Doc 3 (both about inspections)
sim = cosine_similarity([embeddings[2]], [embeddings[5]])[0][0]
print(f"'Do I let landlord inspect?' vs 'Right to conduct inspections...'")
print(f"Similarity: {sim:.4f}\n")

# Cross-topic (should be lower)
sim = cosine_similarity([embeddings[0]], [embeddings[5]])[0][0]
print(f"'Can landlord kick me out?' vs 'Right to conduct inspections...' (different topics)")
print(f"Similarity: {sim:.4f}")