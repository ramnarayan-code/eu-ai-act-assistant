import os
import logging
from typing import Optional

# LlamaIndex imports
from llama_index.core import (
    StorageContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
    load_index_from_storage,
    Settings
)

from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever, AutoMergingRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.vector_stores.simple import SimpleVectorStore

from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
)

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import BaseTool

from llama_index.core.evaluation import generate_question_context_pairs
from llama_index.core.evaluation import RetrieverEvaluator

from traceloop.sdk import Traceloop

from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics import AnswerRelevancy, Faithfulness, FactualCorrectness

import asyncio
from ragas import SingleTurnSample, EvaluationDataset
from ragas import evaluate
from ragas.llms import LlamaIndexLLMWrapper


# Traceloop.init()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class EUIActRAGSystem:
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

        self.evaluator = EUIActRAGSystemEvaluator()
     
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index = None
        self.query_engine = None
        
    def load_and_process_document(
        self, 
        persist_dir: Optional[str] = None
    ):
        """
        Load and process the EU AI Act PDF document
        
        Args:
            pdf_path: Path to the EU AI Act PDF file
            use_semantic: Whether to use semantic chunking
            persist_dir: Directory to persist the index
        """

        #https://developers.llamaindex.ai/python/framework/module_guides/storing/save_load/
        if persist_dir and os.path.exists(persist_dir):
            # Load existing index
            docstore = SimpleDocumentStore.from_persist_path(persist_path=f"{persist_dir}/docstore")
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir, docstore=docstore)

            index = load_index_from_storage(storage_context)

            logger.info("Loaded existing index from storage")
        else:
            logger.info("Index does not exist. Creating a new one is required before querying.")
        
        # Create query engine with enhanced retrieval
        self._create_query_engine(storage_context, index)
        
        logger.info("EU AI Act RAG system ready!")
        
    def _create_query_engine(self, storage_context: StorageContext, base_index):
        """Create an optimized query engine for the EU AI Act"""
        
        # Configure retriever
        # retriever = VectorIndexRetriever(
        #     index=self.index,
        #     similarity_top_k=10,  # Retrieve more candidates initially
        #     embed_model=self.embed_model
        # )

        base_retriever = base_index.as_retriever(similarity_top_k=10)
        retriever = AutoMergingRetriever(base_retriever, storage_context)
        
        # Add post-processors for better results
        # postprocessors = [
        #     SimilarityPostprocessor(similarity_cutoff=0.7)  # Filter low-similarity results
        # ]
        
        
        # Create query engine
        self.query_engine = RetrieverQueryEngine.from_args(retriever, verbose=True)
        
    def get_context(self, query: str) -> str:
        """Retrieve context for a given query"""
        try:
            # Load and process the document
            self.load_and_process_document(
                persist_dir="./eu_ai_act_index"
            )
            context_response = self.query_engine.query(query)
            return context_response.response
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(e)
            print("Please check your OpenAI API key and file paths.")
            return "Error retrieving context."
    
    async def query(
        self, query: str, tools: list[BaseTool] = None
    ) -> str:
        """Query the EU AI Act document with context-aware responses"""
        system_prompt_template = """ Based on the EU AI Act document, 
        please answer the following question based on the context with specific references to 
        relevant articles, chapters, or sections when possible:
        
        Question: {question}
        context: {context}

        Please do not make up any answers. Please do not provide generic answers.
        Please do not use Internet knowledge.
        Please provide a comprehensive answer citing specific parts of the regulation.
        """
        context_response = self.query_engine.query(query)
        context_nodes = [context_response.response]

        print("Context Nodes:   ", context_nodes)

        agent = FunctionAgent(
            tools=tools,
            llm=OpenAI(model="gpt-4o"),
            system_prompt=system_prompt_template.format(context=context_nodes, question=query),
        )

        response = await agent.run(query)

        self.evaluator.evaluate(query, response, context_nodes)
        
        return response
    
class EUIActRAGSystemEvaluator:
    """
    Evaluation module for the EU AI Act RAG system
    """
    def __init__(self):
        self.llm = LlamaIndexLLMWrapper(OpenAI("gpt-4.1"))
        self.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
        )
       

    def evaluate(self, query: str, response: str, context_array: list[str]) -> dict:
        """
        Evaluate the response generated by the RAG system
        
        Args:
            query: The original query
            context_text: The context used for generating the response
            response: The generated response
            
        Returns:
            A dictionary with evaluation metrics
        """
        evaluation_dataset = EvaluationDataset(samples=[SingleTurnSample(
            user_input=query, retrieved_contexts=list(context_array), response=str(response))
        ])

        
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[AnswerRelevancy(), Faithfulness()],
            llm=self.llm
        )


        return result
    
async def main():
    """
    Example usage of the EU AI Act RAG system
    """
    print("Entering...")
    # Configuration
    PDF_PATH = "./data/eu_ai_act.pdf"  # Path to your EU AI Act PDF
    PERSIST_DIR = "./eu_ai_act_index"  # Directory to save/load the index
    
    # Initialize the RAG system
    rag_system = EUIActRAGSystem(
        openai_api_key=OPENAI_API_KEY,
        chunk_size=1024,
        chunk_overlap=200
    )
    
    try:
        # Load and process the document
        rag_system.load_and_process_document(
            persist_dir=PERSIST_DIR
        )
        
        #https://github.com/run-llama/llama_index/issues/12603
        # Example queries
        example_queries = [
            # "My AI application collects PII data, is my application high risk? say yes or no. briefly advise",
            # "What is AI Literacy?",
            # "What is Regulation (EU) 2024/1689?",
            # "Summarize the whole document in 5 bullet points",
            "List me top 3 prohibited AI practices?",
            # "List the classification of AI systems as high risk in bullet points",
            # "Difference between EU AI Act and US AI Act"
        ]
        
        print("\n" + "="*60)
        print("EXAMPLE QUERIES AND RESPONSES")
        print("="*60)
        
        for query in example_queries:
            print(f"\nQ: {query}")
            print("-" * 40)
            response = await rag_system.query(query)
            print(f"A: {response}")
            print("\n" + "="*60)
            
     
    except FileNotFoundError:
        import traceback
        traceback.print_exc()
        print(f"Error: PDF file not found at {PDF_PATH}")
        print("Please ensure you have the EU AI Act PDF file in the specified location.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        print("Please check your OpenAI API key and file paths.")

if __name__ == "__main__":

    asyncio.run(main())