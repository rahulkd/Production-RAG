# Standalone Embedding Generation
import httpx
import asyncio
from typing import List, Optional
import os

from dotenv import load_dotenv

# Load JINA_API_KEY (and other vars) from the project-root .env file,
# regardless of the directory the script is run from.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))

class JinaEmbeddingsGenerator:
    """Standalone Jina AI embeddings generator."""

    def __init__(self, api_key: Optional[str] = None, model: str = "jina-embeddings-v3"):
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        self.model = model
        self.base_url = "https://api.jina.ai/v1/embeddings"
        self.embedding_dimension = 1024
        
        if not self.api_key:
            print("Warning: No Jina API key found. Using dummy embeddings.")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not self.api_key:
            # Return dummy embeddings for demonstration
            return [[0.1] * self.embedding_dimension for _ in texts]
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "input": texts,
            "task": "retrieval.passage"
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                
                result = response.json()
                embeddings = [item["embedding"] for item in result["data"]]
                return embeddings
                
            except Exception as e:
                print(f"Error generating embeddings: {e}")
                return [[0.1] * self.embedding_dimension for _ in texts]

# Test embedding generation
async def main():
    print("TESTING EMBEDDING GENERATION")
    print("=" * 40)

    # API key is read from the JINA_API_KEY environment variable (see __init__).
    embeddings_generator = JinaEmbeddingsGenerator()

    # Test with simple example
    test_texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are computational models inspired by biology."
    ]

    embeddings = await embeddings_generator.generate_embeddings(test_texts)
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Dimension: {len(embeddings[0]) if embeddings else 0}")


if __name__ == "__main__":
    asyncio.run(main())