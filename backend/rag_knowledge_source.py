import os
import logging
import traceback

from typing import Optional

# LlamaIndex imports
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class RAGKnowledgeSource:
    """
    Knowledge source specifically designed for the RAG Use case of EU AI Act document with context-aware chunking
    """

    def __init__(
        self,
        openai_api_key: str,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
        persist_dir: Optional[str] = None,
    ):
        """
        Initialize the RAG system

        Args:
            openai_api_key: OpenAI API key
            embedding_model: OpenAI embedding model to use
            llm_model: OpenAI LLM model to use
            persist_dir: Directory to persist the index
        """

        # Initialize embedding model
        self.embed_model = OpenAIEmbedding(
            model=embedding_model, api_key=openai_api_key
        )

        # Initialize LLM
        self.llm = OpenAI(model=llm_model, api_key=openai_api_key, temperature=0.1)

        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        self.__load_rag_index(persist_dir)

    def __load_rag_index(self, persist_dir: Optional[str] = None):
        """
        Load or create the index for the RAG

        Args:
            persist_dir: Directory to persist the index
        """

        # https://developers.llamaindex.ai/python/framework/module_guides/storing/save_load/
        if persist_dir and os.path.exists(persist_dir):
            # Load existing index
            docstore = SimpleDocumentStore.from_persist_path(
                persist_path=f"{persist_dir}/docstore"
            )
            storage_context = StorageContext.from_defaults(
                persist_dir=persist_dir, docstore=docstore
            )

            self.index = load_index_from_storage(storage_context)
            logger.info("Loaded existing index from storage")
        else:
            logger.info(
                "Index does not exist. Creating a new one is required before querying."
            )

        # Create query engine with enhanced retrieval
        self.query_engine = self.__create_query_engine(storage_context, self.index)

        logger.info("Index for RAG is ready!")

    def __create_query_engine(
        self, storage_context: StorageContext, base_index
    ) -> RetrieverQueryEngine:
        """
        Create an optimized query engine for the EU AI Act

        Args:
            storage_context: The storage context containing the index
            base_index: The base index to create the retriever from
        Returns:
            An optimized RetrieverQueryEngine instance
        """
        return RetrieverQueryEngine.from_args(
            AutoMergingRetriever(
                base_index.as_retriever(similarity_top_k=10), storage_context
            ),
            verbose=True,
        )

    def get_context(self, query: str) -> str:
        """Retrieve context for a given query

        Args:
            query: The user query
        Returns:
            The retrieved context as a string"""
        try:
            # Load and process the document
            context_response = self.query_engine.query(query)
            return context_response.response
        except Exception as e:
            traceback.print_exc()
            logger.error("Please check your OpenAI API key and file paths.")
            return "Error retrieving context."
