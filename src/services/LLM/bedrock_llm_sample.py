import boto3
import json
import os

from dotenv import load_dotenv
from botocore.exceptions import ClientError

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
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

model_id = "meta.llama3-70b-instruct-v1:0"

prompt = "Please explain the concept of Retrieval-Augmented Generation (RAG) in the context of natural language processing and how it can be applied to improve the performance of language models."

formatted_prompt = f"""
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

native_request = {
    "prompt": formatted_prompt,
    "max_gen_len": 512,
    "temperature": 0.5,
}

request = json.dumps(native_request)

try:
    response = client.invoke_model_with_response_stream(modelId=model_id, body=request)
except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)

for event in response["body"]:
    chunk = json.loads(event["chunk"]["bytes"])
    if chunk.get("generation"):
        print(chunk["generation"], end="", flush=True)

print()
