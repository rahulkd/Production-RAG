import boto3
import json
import os

from dotenv import load_dotenv
from botocore.exceptions import ClientError

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))

bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
if not bearer_token:
    raise EnvironmentError("AWS_BEARER_TOKEN_BEDROCK not set in .env")

# botocore reads AWS_BEARER_TOKEN_BEDROCK from the environment and uses bearer
# auth for bedrock/bedrock-runtime; it is not a SigV4 session token.
client = boto3.client(
    "bedrock-runtime",
    region_name="us-west-2",
)

model_id = "qwen.qwen3-vl-235b-a22b"  ## 256k context, 235B parameters, instruction-tuned LLM with vision capabilities

prompt = "Please explain the concept of Retrieval-Augmented Generation (RAG) in the context of natural language processing and how it can be applied to improve the performance of language models."

# Qwen on Bedrock uses an OpenAI-compatible schema: a `messages` array and
# `max_tokens`. This is why the Llama-style body ({"prompt": ..., "max_gen_len": ...})
# fails with 'missing field `messages`' - there is no shared native format
# across model families on invoke_model.
native_request = {
    "messages": [
        {"role": "user", "content": prompt},
    ],
    "max_tokens": 512,
    "temperature": 0.5,
}

request = json.dumps(native_request)

try:
    response = client.invoke_model_with_response_stream(modelId=model_id, body=request)
except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)

# Qwen streams OpenAI-style chunks: {"choices": [{"delta": {"content": "..."},
# "finish_reason": null}], ...}
for event in response["body"]:
    chunk = json.loads(event["chunk"]["bytes"])

    for choice in chunk.get("choices", []):
        text = choice.get("delta", {}).get("content")
        if text:
            print(text, end="", flush=True)

        if choice.get("finish_reason"):
            break

print()
