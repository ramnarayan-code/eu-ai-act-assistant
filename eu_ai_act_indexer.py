import os
import logging

from typing import Optional

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.node_parser import (
    SentenceSplitter,
    HierarchicalNodeParser,
    get_leaf_nodes,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.extractors import (
    TitleExtractor,
    SummaryExtractor,
    KeywordExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.schema import Document

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EUIActIndexer:
    """
    Indexer for the EU AI Act document
    """

    def __init__(
        self,
        openai_api_key: str,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
    ):
        """
        Args:
            openai_api_key: OpenAI API key
            embedding_model: OpenAI embedding model to use
            llm_model: OpenAI LLM model to use
        """
        # Initialize embedding model
        self.embed_model = OpenAIEmbedding(
            model=embedding_model, api_key=openai_api_key
        )

        # Initialize LLM
        self.llm = OpenAI(model=llm_model, api_key=openai_api_key, temperature=0.1)

        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

    def __create_chunker(self) -> HierarchicalNodeParser:
        """
        Create a hierarchical node parser that maintains document context
        This strategy preserves topic coherence in the EU AI Act

        Returns:
            HierarchicalNodeParser: The created hierarchical node parser
        """
        # Primary splitter for large sections (articles, chapters)
        large_splitter = SentenceSplitter(
            chunk_size=2048,
            chunk_overlap=100,
            separator="\n\n",  # Split on paragraph breaks
        )

        # Secondary splitter for smaller, coherent chunks
        small_splitter = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=100,
            separator=". ",  # Split on sentences
        )

        # Create hierarchical parser
        hierarchical_parser = HierarchicalNodeParser.from_defaults(
            node_parser_map={
                "large_splitter": large_splitter,
                "small_splitter": small_splitter,
            },
            include_metadata=True,
            include_prev_next_rel=True,  # Maintains context relationships
        )

        return hierarchical_parser

    def __create_enhanced_pipeline(self) -> IngestionPipeline:
        """
        Create an ingestion pipeline with metadata extractors for better context

        Returns:
            IngestionPipeline: The created ingestion pipeline
        """

        node_parser = self.__create_chunker()

        # Create extractors for enhanced metadata
        extractors = [
            TitleExtractor(nodes=5, llm=self.llm),  # Extract titles/headings
            KeywordExtractor(keywords=10, llm=self.llm),  # Extract key terms
            SummaryExtractor(
                summaries=["prev", "self"], llm=self.llm
            ),  # Context summaries
        ]

        # Create ingestion pipeline
        pipeline = IngestionPipeline(
            transformations=[node_parser] + extractors + [self.embed_model]
        )

        return pipeline

    def __create_vector_store_index(self, nodes) -> VectorStoreIndex:
        """
        Create a vector store index from the given nodes

        Args:
            nodes: List of document nodes to index

        Returns:
            VectorStoreIndex: The created vector store index
        """
        return VectorStoreIndex(
            nodes=nodes,
            show_progress=True,
        )

    def __load_document(
        self, pdf_path: str, start_end_page_numbers: Optional[tuple[int, int]] = None
    ) -> list[Document]:
        """
        Load the EU AI Act PDF document

        Args:
            pdf_path: Path to the EU AI Act PDF file
            start_end_page_numbers: Start and end page numbers for partial loading
        Returns:
            List[Document]: Loaded document(s)
        """
        # Load the PDF document
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Loading document from: {pdf_path}")

        # Use SimpleDirectoryReader for PDF parsing
        reader = SimpleDirectoryReader(input_files=[pdf_path], required_exts=[".pdf"])
        documents = reader.load_data()
        total_pages = len(documents)

        if start_end_page_numbers:
            start = start_end_page_numbers[0]
            end = start_end_page_numbers[1]
            documents = documents[start:end]

            logger.info(
                f"Loaded {len(documents)} document(s) partially from pages {start} to {end} out of {total_pages} total pages"
            )
        else:
            logger.info(f"Loaded the complete {total_pages} document(s)")

        # Add custom metadata for EU AI Act context
        for doc in documents:
            doc.metadata.update(
                {
                    "document_type": "EU_AI_Act",
                    "source": "European Union",
                    "document_category": "Legal_Regulation",
                    "topic_domain": "Artificial_Intelligence",
                }
            )

        return documents

    def index_document(
        self,
        pdf_path: str,
        start_end_page_numbers: Optional[tuple[int, int]] = None,
        persist_dir: Optional[str] = None,
    ):
        """
        Index the EU AI Act PDF document

        Args:
            pdf_path: Path to the EU AI Act PDF file
            start_end_page_numbers: Start and end page numbers for partial loading
            persist_dir: Directory to persist the index
        """

        if persist_dir and os.path.exists(persist_dir):
            # Load existing index
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            self.index = load_index_from_storage(storage_context)
            logger.info("Loaded existing index from storage")
        else:
            documents = self.__load_document(
                pdf_path, start_end_page_numbers=start_end_page_numbers
            )

            # Create processing pipeline
            pipeline = self.__create_enhanced_pipeline()

            # Process documents through pipeline
            logger.info("Processing document through ingestion pipeline...")

            all_nodes = pipeline.run(documents=documents, show_progress=True)
            leaf_nodes = get_leaf_nodes(all_nodes)

            logger.info(f"Created {len(leaf_nodes)} chunks/nodes for indexing")

            index = self.__create_vector_store_index(leaf_nodes)

            # Persist index if directory specified
            if persist_dir:
                index.storage_context.persist(persist_dir=persist_dir)

                docstore = SimpleDocumentStore()
                docstore.add_documents(all_nodes)
                docstore.persist(persist_path=f"{persist_dir}/docstore")

                logger.info(f"Index persisted to: {persist_dir}")

        logger.info("EU AI Act RAG system ready!")


def main():
    # Configuration
    PDF_PATH = "./data/eu_ai_act.pdf"  # Path to your EU AI Act PDF
    PERSIST_DIR = "./eu_ai_act_index_test"  # Directory to save/load the index

    # Initialize the RAG system
    rag_system = EUIActIndexer(openai_api_key=OPENAI_API_KEY)

    # Load and process the document
    try:
        rag_system.index_document(
            pdf_path=PDF_PATH,
            start_end_page_numbers=(0, 30),  # Example page range
            persist_dir=PERSIST_DIR,
        )

        logger.info("EU AI Act document indexed successfully.")
    except FileNotFoundError:
        logger.error(f"Error: PDF file not found at {PDF_PATH}")
        logger.error(
            "Please ensure you have the EU AI Act PDF file in the specified location."
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error("Please check your OpenAI API key and file paths.")


if __name__ == "__main__":
    main()
