import os
import logging
from typing import Optional

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
    HierarchicalNodeParser,
    get_leaf_nodes
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.extractors import (
    TitleExtractor,
    SummaryExtractor,
    KeywordExtractor
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore

import asyncio

# Traceloop.init()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class EUIActIndexer:
    """
    RAG system specifically designed for the EU AI Act document with context-aware chunking
    """
    
    def __init__(
        self,
        openai_api_key: str,
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini"
    ):
        """
        Initialize the RAG system
        
        Args:
            openai_api_key: OpenAI API key
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embedding_model: OpenAI embedding model to use
            llm_model: OpenAI LLM model to use
        """
        # Set OpenAI API key
        # openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize embedding model
        self.embed_model = OpenAIEmbedding(
            model=embedding_model,
            api_key=openai_api_key
        )
        
        # Initialize LLM
        self.llm = OpenAI(
            model=llm_model,
            api_key=openai_api_key,
            temperature=0.1
        )
     
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index = None
        self.query_engine = None
        
    def create_context_aware_chunker(self) -> HierarchicalNodeParser:
        """
        Create a hierarchical node parser that maintains document context
        This strategy preserves topic coherence in the EU AI Act
        """
        # Primary splitter for large sections (articles, chapters)
        large_splitter = SentenceSplitter(
            chunk_size=2048,
            chunk_overlap=100,
            separator="\n\n"  # Split on paragraph breaks
        )
        
        # Secondary splitter for smaller, coherent chunks
        small_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=". "  # Split on sentences
        )
        
    
        # Create hierarchical parser
        hierarchical_parser = HierarchicalNodeParser.from_defaults(
            node_parser_map={"large_splitter": large_splitter, "small_splitter": small_splitter},
            include_metadata=True,
            include_prev_next_rel=True  # Maintains context relationships
        )

        return hierarchical_parser
        
    def create_semantic_chunker(self) -> SemanticSplitterNodeParser:
        """
        Alternative: Semantic chunking that groups semantically similar content
        Good for maintaining topic coherence in legal documents
        """
        semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=1,  # Number of sentences to group
            embed_model=self.embed_model,
            breakpoint_percentile_threshold=95,  # Threshold for semantic breaks
            include_metadata=True
        )
        
        return semantic_splitter
        
    def create_enhanced_pipeline(self, use_semantic: bool = False):
        """
        Create an ingestion pipeline with metadata extractors for better context
        
        Args:
            use_semantic: Whether to use semantic chunking instead of hierarchical
        """
        # Choose chunking strategy
        if use_semantic:
            node_parser = self.create_semantic_chunker()
        else:
            node_parser = self.create_context_aware_chunker()
        
        # Create extractors for enhanced metadata
        extractors = [
            TitleExtractor(nodes=5, llm=self.llm),  # Extract titles/headings
            KeywordExtractor(keywords=10, llm=self.llm),  # Extract key terms
            SummaryExtractor(summaries=["prev", "self"], llm=self.llm),  # Context summaries
        ]
        
        # Create ingestion pipeline
        pipeline = IngestionPipeline(
            transformations=[node_parser] + extractors + [self.embed_model]
        )
        
        return pipeline
        
    def index_document(
        self, 
        pdf_path: str, 
        start_end_page_numbers: Optional[tuple[int, int]] = None,
        use_semantic: bool = False,
        persist_dir: Optional[str] = None
    ):
        """
        Load and index the EU AI Act PDF document
        
        Args:
            pdf_path: Path to the EU AI Act PDF file
            use_semantic: Whether to use semantic chunking
            persist_dir: Directory to persist the index
        """

        if persist_dir and os.path.exists(persist_dir):
            # Load existing index
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            self.index = load_index_from_storage(
                storage_context,
                # service_context=self.service_context
            )
            logger.info("Loaded existing index from storage")
        else:
            logger.info(f"Loading document from: {pdf_path}")
            
            # Load the PDF document
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                
            # Use SimpleDirectoryReader for PDF parsing
            reader = SimpleDirectoryReader(
                input_files=[pdf_path],
                required_exts=[".pdf"]
            )
            
            documents = reader.load_data()
            total_pages = len(documents)

            if start_end_page_numbers:
                start = start_end_page_numbers[0]
                end = start_end_page_numbers[1]
                documents = documents[start:end]

                logger.info(f"Loaded {len(documents)} document(s) partially from pages {start} to {end} out of {total_pages} total pages")
            else:
                logger.info(f"Loaded the complete {total_pages} document(s)")
                    
            # Add custom metadata for EU AI Act context
            for doc in documents:
                doc.metadata.update({
                    "document_type": "EU_AI_Act",
                    "source": "European Union",
                    "document_category": "Legal_Regulation",
                    "topic_domain": "Artificial_Intelligence"
                })            
            
            # Create processing pipeline
            pipeline = self.create_enhanced_pipeline(use_semantic=use_semantic)
            
            # Process documents through pipeline
            logger.info("Processing document through ingestion pipeline...")
            all_nodes = pipeline.run(documents=documents, show_progress=True)

            docstore = SimpleDocumentStore()
            docstore.add_documents(all_nodes)

            leaf_nodes = get_leaf_nodes(all_nodes)
            
            logger.info(f"Created {len(leaf_nodes)} chunks/nodes")
            
            # Create vector store index
            logger.info("Creating vector store index...")
            
            # Create new index
            self.index = VectorStoreIndex(
                nodes=leaf_nodes,
                # service_context=self.service_context,
                show_progress=True
            )
            
            # Persist index if directory specified
            if persist_dir:
                self.index.storage_context.persist(persist_dir=persist_dir)
                docstore.persist(persist_path=f"{persist_dir}/docstore")
                logger.info(f"Index persisted to: {persist_dir}")
            
        logger.info("EU AI Act RAG system ready!")
    
async def main():
    """
    Example usage of the EU AI Act RAG system
    """
    # Configuration
    PDF_PATH = "./data/eu_ai_act.pdf"  # Path to your EU AI Act PDF
    PERSIST_DIR = "./eu_ai_act_index"  # Directory to save/load the index
    
    # Initialize the RAG system
    rag_system = EUIActIndexer(
        openai_api_key=OPENAI_API_KEY,
        chunk_size=512,
        chunk_overlap=200
    )
    
    # Load and process the document
    try:
        # Hierarchical chunking with context-aware strategy performs better for legal documents
        rag_system.index_document(
            pdf_path=PDF_PATH,
            start_end_page_numbers=(0,30),  # Example page range
            use_semantic=False,  # Set to True for semantic chunking
            persist_dir=PERSIST_DIR
        )
        
        # Get document statistics
        stats = rag_system.get_document_stats()
        print("\nDocument Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
            
    except FileNotFoundError:
        print(f"Error: PDF file not found at {PDF_PATH}")
        print("Please ensure you have the EU AI Act PDF file in the specified location.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        print("Please check your OpenAI API key and file paths.")

if __name__ == "__main__":

    asyncio.run(main())