from functools import lru_cache
from typing import Annotated, Generator

from fastapi import Depends, Request
from sqlalchemy.orm import Session
from src.config import Settings
from src.db.interfaces.base import BaseDatabase
from src.services.arxiv.client import ArxivClient
from src.services.opensearch.client import OpenSearchClient
from src.services.pdf_parser.parser import PDFParserService
## embedding part
#from src.services.embeddings.jina_client import JinaEmbeddingsClient

## why does this file exist?
## This file defines the dependencies for the application, such as database connections, clients for external services, and application
#  settings. 
## It provides a centralized place to manage these dependencies and makes it easy to inject them into FastAPI routes and other parts of
#  the application using the Depends mechanism. 
## By using dependency injection, we can keep our code modular and testable, as we can easily swap out implementations of these 
# dependencies for testing or different environments.


from functools import lru_cache
from typing import Annotated, Generator

from fastapi import Depends, Request
from sqlalchemy.orm import Session
from src.config import Settings
from src.db.interfaces.base import BaseDatabase
from src.services.arxiv.client import ArxivClient
from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.opensearch.client import OpenSearchClient
from src.services.pdf_parser.parser import PDFParserService
from src.services.LLM.client import BedrockClient
## from src.services.LLM.client_ollama import OllamaClient ## ollama


@lru_cache
def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


def get_request_settings(request: Request) -> Settings:
    """Get settings from the request state."""
    return request.app.state.settings


def get_database(request: Request) -> BaseDatabase:
    """Get database from the request state."""
    return request.app.state.database


def get_db_session(database: Annotated[BaseDatabase, Depends(get_database)]) -> Generator[Session, None, None]:
    """Get database session dependency."""
    with database.get_session() as session:
        yield session


def get_opensearch_client(request: Request) -> OpenSearchClient:
    """Get OpenSearch client from the request state."""
    return request.app.state.opensearch_client


def get_arxiv_client(request: Request) -> ArxivClient:
    """Get arXiv client from the request state."""
    return request.app.state.arxiv_client


def get_pdf_parser(request: Request) -> PDFParserService:
    """Get PDF parser service from the request state."""
    return request.app.state.pdf_parser


def get_embeddings_service(request: Request) -> JinaEmbeddingsClient:
    """Get embeddings service from the request state."""
    return request.app.state.embeddings_service


def get_bedrock_client(request: Request) -> BedrockClient:
    """Get Bedrock LLM client from the request state."""
    return request.app.state.bedrock_client


## ollama
#def get_ollama_client(request: Request) -> OllamaClient:
#    """Get Ollama client from the request state."""
#    return request.app.state.ollama_client


# Dependency annotations
SettingsDep = Annotated[Settings, Depends(get_settings)]
DatabaseDep = Annotated[BaseDatabase, Depends(get_database)]
SessionDep = Annotated[Session, Depends(get_db_session)]
OpenSearchDep = Annotated[OpenSearchClient, Depends(get_opensearch_client)]
ArxivDep = Annotated[ArxivClient, Depends(get_arxiv_client)]
PDFParserDep = Annotated[PDFParserService, Depends(get_pdf_parser)]
EmbeddingsDep = Annotated[JinaEmbeddingsClient, Depends(get_embeddings_service)]
BedrockDep = Annotated[BedrockClient, Depends(get_bedrock_client)]
#OllamaDep = Annotated[OllamaClient, Depends(get_ollama_client)]  ## ollama