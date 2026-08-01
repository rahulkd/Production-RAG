# Test Complete RAG Pipeline with STREAMING
import json
import os
import time

import requests

API_BASE = os.getenv("RAG_API_BASE", "http://localhost:8000/api/v1")

# Bedrock model id (was "llama3.2:1b" when this ran on local Ollama).
MODEL = os.getenv("BEDROCK_MODEL", "meta.llama3-70b-instruct-v1:0")

print("COMPLETE RAG PIPELINE TEST (STREAMING)")
print("=" * 40)

question = "explain poisoning of pretrained data using external web sources"
print(f"Question: {question}")

start_time = time.time()

try:
    rag_request = {
        "query": question,
        "top_k": 3,  # Use 1 chunk for context
        "use_hybrid": True,  # Use best search
        "model": MODEL
    }

    # Using streaming endpoint for real-time responses
    response = requests.post(
        f"{API_BASE}/stream",
        json=rag_request,
        stream=True,  # Enable streaming
        timeout=120  # Bedrock 70B is slower than a local 1B model
    )

    if response.status_code == 200:
        # Process streaming response
        full_answer = ""
        sources = []
        chunks_used = 0
        search_mode = "unknown"
        first_chunk_time = None
        error = None

        print(f"\nStreaming response...")

        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    try:
                        data = json.loads(line_str[6:])  # Remove 'data: ' prefix

                        # Handle server-side errors reported inside the stream
                        if 'error' in data:
                            error = data['error']
                            break

                        # Handle metadata
                        if 'sources' in data:
                            sources = data['sources']
                            chunks_used = data.get('chunks_used', 0)
                            search_mode = data.get('search_mode', 'unknown')

                        # Handle streaming chunks
                        if 'chunk' in data:
                            if first_chunk_time is None:
                                first_chunk_time = time.time() - start_time
                                print(f"First response in: {first_chunk_time:.1f} seconds")
                                print("\nAnswer:")
                                print("-" * 40)

                            chunk_text = data['chunk']
                            full_answer += chunk_text
                            print(chunk_text, end='', flush=True)  # Print as it streams

                        # Handle completion
                        if data.get('done', False):
                            break

                    except json.JSONDecodeError:
                        continue

        response_time = time.time() - start_time

        if error:
            print(f"\n✗ Stream reported an error: {error}")
        else:
            print("\n" + "-" * 40)
            print(f"\n✓ Complete! (Total: {response_time:.1f} seconds)")

            print(f"\nSources: {len(sources)} papers")
            if sources:
                for i, source in enumerate(sources[:2], 1):
                    print(f"  {i}. {source}")
            print(f"Chunks used: {chunks_used}")
            print(f"Search mode: {search_mode}")

    else:
        print(f"\n✗ Request failed: HTTP {response.status_code}")
        print(f"Response: {response.text[:200]}")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
