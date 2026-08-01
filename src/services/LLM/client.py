"""AWS Bedrock LLM client.

Drop-in replacement for the local Ollama client (kept, commented out, in
``client_ollama.py``). The public surface is deliberately identical so callers
need only swap the injected dependency:

    health_check / list_models / generate / generate_stream /
    generate_rag_answer / generate_rag_answer_stream

Generation goes through the Bedrock **Converse** API rather than
``invoke_model``. Converse normalises messages and inference parameters across
model families, so ``model`` can be any Bedrock model id the account has access
to without per-family prompt templating.

boto3 is synchronous; every call is offloaded with ``asyncio.to_thread`` so the
async interface never blocks the event loop.
"""

import asyncio
import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, ConnectTimeoutError, EndpointConnectionError, ReadTimeoutError
from src.config import Settings
from src.exceptions import (
    BedrockAuthenticationError,
    BedrockConnectionError,
    BedrockException,
    BedrockThrottlingError,
    BedrockTimeoutError,
)
from src.services.LLM.prompts import RAGPromptBuilder, ResponseParser

logger = logging.getLogger(__name__)

_AUTH_ERROR_CODES = {
    "AccessDeniedException",
    "UnrecognizedClientException",
    "ExpiredTokenException",
    "InvalidSignatureException",
}


class BedrockClient:
    """Client for interacting with the AWS Bedrock runtime."""

    def __init__(self, settings: Settings):
        """Initialize Bedrock client with settings."""
        config = settings.bedrock

        self.region_name = config.region_name
        self.default_model = config.model_id
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.top_p = config.top_p

        # botocore reads the Bedrock API key from the process environment and
        # switches to bearer auth. pydantic-settings loads .env into Settings
        # only, so export it here when the process itself was not given one.
        if settings.aws_bearer_token_bedrock:
            os.environ.setdefault("AWS_BEARER_TOKEN_BEDROCK", settings.aws_bearer_token_bedrock)

        if not os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
            logger.warning(
                "AWS_BEARER_TOKEN_BEDROCK is not set; Bedrock calls will fall back to the default AWS credential chain"
            )

        ## explain botoconfig: The BotoConfig object is used to configure the behavior of the boto3 client. 
        # It allows you to set various parameters such as timeouts and retry strategies. 
        # In this case, we are setting the connect timeout, read timeout, and 
        # maximum number of retries for the Bedrock client. 
        # This ensures that the client behaves as expected when making requests to the Bedrock service, 
        # handling potential network issues gracefully.

        boto_config = BotoConfig(
            connect_timeout=config.connect_timeout,
            read_timeout=config.read_timeout,
            retries={"max_attempts": config.max_retries, "mode": "standard"},
        )

        self.client = boto3.client("bedrock-runtime", region_name=self.region_name, config=boto_config)
        # Control plane, used for health checks and model listing only.
        self.control_client = boto3.client("bedrock", region_name=self.region_name, config=boto_config)

        self.prompt_builder = RAGPromptBuilder()
        self.response_parser = ResponseParser()

    @staticmethod
    def _wrap_error(error: Exception, context: str) -> BedrockException:
        """Translate a botocore error into the project's exception hierarchy.

        Args:
            error: Original exception raised by boto3/botocore
            context: Short description of the operation that failed

        Returns:
            A BedrockException subclass appropriate to the failure
        """
        if isinstance(error, (ReadTimeoutError, ConnectTimeoutError)):
            return BedrockTimeoutError(f"{context}: Bedrock request timed out: {error}")

        if isinstance(error, EndpointConnectionError):
            return BedrockConnectionError(f"{context}: cannot reach Bedrock endpoint: {error}")

        if isinstance(error, ClientError):
            code = error.response.get("Error", {}).get("Code", "")
            if code in _AUTH_ERROR_CODES:
                return BedrockAuthenticationError(f"{context}: Bedrock rejected the API key ({code}): {error}")
            if code == "ThrottlingException":
                return BedrockThrottlingError(f"{context}: Bedrock throttled the request: {error}")
            return BedrockException(f"{context}: Bedrock returned {code or 'an error'}: {error}")

        return BedrockException(f"{context}: {error}")

    def _build_inference_config(self, **kwargs) -> Dict[str, Any]:
        """Map generation kwargs onto a Converse ``inferenceConfig`` block.

        Accepts the Ollama/Llama spellings (``max_gen_len``, ``num_predict``)
        alongside ``max_tokens`` so existing call sites keep working unchanged.
        """
        max_tokens = kwargs.get("max_tokens", kwargs.get("max_gen_len", kwargs.get("num_predict", self.max_tokens)))
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)

        inference_config: Dict[str, Any] = {
            "maxTokens": int(max_tokens),
            "temperature": float(temperature),
            "topP": float(top_p),
        }

        stop_sequences = kwargs.get("stop", kwargs.get("stop_sequences"))
        if stop_sequences:
            inference_config["stopSequences"] = list(stop_sequences)

        return inference_config

    async def health_check(self) -> Dict[str, Any]:
        """
        Check if Bedrock is reachable and the API key is accepted.

        Returns:
            Dictionary with health status information
        """
        try:
            response = await asyncio.to_thread(self.control_client.list_foundation_models)
            models = response.get("modelSummaries", [])

            return {
                "status": "healthy",
                "message": f"Bedrock reachable in {self.region_name}",
                "region": self.region_name,
                "model": self.default_model,
                "available_models": len(models),
            }

        except BedrockException:
            raise
        except Exception as e:
            raise self._wrap_error(e, "Bedrock health check failed")

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        Get list of foundation models available to this account/region.

        Returns:
            List of model information dictionaries
        """
        try:
            response = await asyncio.to_thread(self.control_client.list_foundation_models)

            return [
                {
                    "name": summary.get("modelId"),
                    "model": summary.get("modelId"),
                    "provider": summary.get("providerName"),
                    "input_modalities": summary.get("inputModalities", []),
                    "output_modalities": summary.get("outputModalities", []),
                    "streaming": summary.get("responseStreamingSupported", False),
                }
                for summary in response.get("modelSummaries", [])
            ]

        except BedrockException:
            raise
        except Exception as e:
            raise self._wrap_error(e, "Error listing Bedrock models")

    async def generate(
        self,
        model: Optional[str] = None,
        prompt: str = "",
        stream: bool = False,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate text using the specified Bedrock model.

        Args:
            model: Bedrock model id (defaults to the configured model)
            prompt: Input prompt for generation
            stream: Kept for interface parity; use generate_stream() to stream
            **kwargs: Additional generation parameters (temperature, top_p, max_tokens, ...)

        Returns:
            Response dictionary shaped like the Ollama client's, with the
            generated text under ``response``
        """
        model_id = model or self.default_model
        inference_config = self._build_inference_config(**kwargs)

        try:
            logger.info(f"Sending request to Bedrock: model={model_id}, inference_config={inference_config}")

            ## explain asyncio.to_thread: The asyncio.to_thread function is used to run a synchronous function in a separate thread, 
            # allowing the main event loop to continue running without blocking.
            response = await asyncio.to_thread(
                self.client.converse,
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig=inference_config,
            )

            content_blocks = response.get("output", {}).get("message", {}).get("content", [])
            text = "".join(block.get("text", "") for block in content_blocks)

            return {
                "model": model_id,
                "response": text,
                "done": True,
                "stop_reason": response.get("stopReason"),
                "usage": response.get("usage", {}),
            }

        except BedrockException:
            raise
        except Exception as e:
            raise self._wrap_error(e, f"Error generating with Bedrock model '{model_id}'")


    ## explain how the generate_stream function works: The generate_stream function is an asynchronous generator that allows for 
    # streaming text generation from a specified Bedrock model. 
    # It takes in a model ID, a prompt, and additional generation parameters. 
    # he function uses the Bedrock client's converse_stream method to send the prompt to the model and receive a streaming response. 
    # It processes the events from the response stream, yielding chunks of generated text as they are received. 
    # The function also handles stop reasons and usage metadata, yielding a final chunk indicating completion when the stream ends.
    #  This allows for real-time text generation without blocking the main event loop.
    
    ## so basically generate_stream is an async generator that streams text generation from a Bedrock model, 
    # yielding chunks of generated text in real-time while handling stop reasons and usage metadata.
    async def generate_stream(self, model: Optional[str] = None, prompt: str = "", **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """
        Generate text with a streaming response.

        Args:
            model: Bedrock model id (defaults to the configured model)
            prompt: Input prompt for generation
            **kwargs: Additional generation parameters

        Yields:
            Chunks shaped like Ollama's streaming payloads -- ``{"response":
            ..., "done": bool}`` -- so downstream consumers need no changes.
        """
        model_id = model or self.default_model
        inference_config = self._build_inference_config(**kwargs)

        try:
            logger.info(f"Starting streaming generation: model={model_id}")

            response = await asyncio.to_thread(
                self.client.converse_stream,
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig=inference_config,
            )

            # The event stream is a blocking iterator; step it off-thread so a
            # slow model cannot stall the event loop.
            ## explain iter(response["stream"]): The iter(response["stream"]) creates an iterator over the streaming response 
            # from the Bedrock model.
            events = iter(response["stream"])
            sentinel = object()
            stop_reason = None
            usage: Dict[str, Any] = {}

            while True:
                ## explain asyncio.to_thread(next, events, sentinel): The asyncio.to_thread(next, events, sentinel) is used to 
                # run the next() function in a separate thread, allowing the main event loop to continue running without blocking.
                event = await asyncio.to_thread(next, events, sentinel)
                if event is sentinel:
                    break

                if "contentBlockDelta" in event:
                    text = event["contentBlockDelta"].get("delta", {}).get("text", "")
                    if text:
                        yield {"model": model_id, "response": text, "done": False}

                elif "messageStop" in event:
                    stop_reason = event["messageStop"].get("stopReason")

                elif "metadata" in event:
                    usage = event["metadata"].get("usage", {})

            # Terminal chunk mirrors Ollama's final `done` message.
            yield {"model": model_id, "response": "", "done": True, "stop_reason": stop_reason, "usage": usage}

        except BedrockException:
            raise
        except Exception as e:
            raise self._wrap_error(e, f"Error in streaming generation with Bedrock model '{model_id}'")

    async def generate_rag_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: Optional[str] = None,
        use_structured_output: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a RAG answer using retrieved chunks.

        Args:
            query: User's question
            chunks: Retrieved document chunks with metadata
            model: Bedrock model id to use for generation
            use_structured_output: Ask the model for schema-conforming JSON

        Returns:
            Dictionary with answer, sources, confidence, and citations
        """
        model_id = model or self.default_model

        try:
            if use_structured_output:
                # Converse has no schema parameter, so the schema lives in the prompt.
                prompt = self.prompt_builder.create_bedrock_structured_prompt(query, chunks)
            else:
                prompt = self.prompt_builder.create_rag_prompt(query, chunks)

            response = await self.generate(
                model=model_id,
                prompt=prompt,
                temperature=0.7,
                top_p=0.9,
            )

            if response and response.get("response"):
                # Parse the LLM response
                logger.debug(f"Raw LLM response: {response['response'][:500]}")
                parsed_response = self.response_parser.parse_structured_response(response["response"])
                logger.debug(f"Parsed response: {parsed_response}")

                # Ensure sources are included if not already
                if not parsed_response.get("sources"):
                    # Build PDF URLs from arxiv_ids
                    sources = []
                    seen_urls = set()
                    for chunk in chunks:
                        arxiv_id = chunk.get("arxiv_id")
                        if arxiv_id:
                            # Build PDF URL from arxiv_id
                            arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                            pdf_url = f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf"
                            if pdf_url not in seen_urls:
                                sources.append(pdf_url)
                                seen_urls.add(pdf_url)
                    parsed_response["sources"] = sources

                # Add citations if not present
                if not parsed_response.get("citations"):
                    # Extract unique arxiv IDs
                    citations = list(set(chunk.get("arxiv_id") for chunk in chunks if chunk.get("arxiv_id")))
                    parsed_response["citations"] = citations[:5]  # Limit to 5 citations

                return parsed_response
            else:
                raise BedrockException("No response generated from Bedrock")

        except BedrockException:
            raise
        except Exception as e:
            logger.error(f"Error generating RAG answer: {e}")
            raise BedrockException(f"Failed to generate RAG answer: {e}")

    async def generate_rag_answer_stream(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Generate a streaming RAG answer using retrieved chunks.

        Args:
            query: User's question
            chunks: Retrieved document chunks with metadata
            model: Bedrock model id to use for generation

        Yields:
            Streaming response chunks with partial answers
        """
        model_id = model or self.default_model

        try:
            # Create prompt for streaming (plain text, not the JSON-schema variant)
            prompt = self.prompt_builder.create_rag_prompt(query, chunks)

            # Stream the response
            async for chunk in self.generate_stream(
                model=model_id,
                prompt=prompt,
                temperature=0.7,
                top_p=0.9,
            ):
                yield chunk

        except BedrockException:
            raise
        except Exception as e:
            logger.error(f"Error generating streaming RAG answer: {e}")
            raise BedrockException(f"Failed to generate streaming RAG answer: {e}")
