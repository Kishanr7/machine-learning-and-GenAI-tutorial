import logging
from datetime import datetime
import os
from IPython.display import Markdown, display
from llama_index.llms.openai import OpenAI
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, StorageContext,
    Settings, ServiceContext
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, PodSpec

# Setup logging
def setup_logging():
    try:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"../../GenAiParserLogs/log_PC_{current_time}.log"
        logging.basicConfig(filename=log_filename,
                            level=logging.DEBUG,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    except Exception as e:
        logging.error(f"Error in setting up logging: {e}")

# Initialize LLM Service
def initialize_llm_service():
    try:
        Settings.llm = OpenAI(model='gpt-3.5-turbo', temperature=0, max_tokens=256)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
        Settings.chunk_size = 1000
    except Exception as e:
        logging.error(f"Error in initializing LLM service: {e}")
    return Settings

# Load and preprocess documents
def load_and_preprocess_documents(path):
    try:
        documents = SimpleDirectoryReader(path).load_data()
        logging.info("Documents loaded and preprocessed.")
        return documents
    except Exception as e:
        logging.error(f"Error in loading and preprocessing documents: {e}")

# Create and persist index
def create_and_persist_index(documents):
    try:
        pc = Pinecone(
            api_key=os.environ.get('PINECONE_API_KEY')
        )
        index_name = 'first-llama-index'
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                index_name,
                dimension=1536,
                metric='cosine',
                spec=PodSpec(
                    environment="gcp-starter"
                )
            )
            
        pinecone_index = pc.Index(index_name)
        
        vector_store =  PineconeVectorStore(pinecone_index=pinecone_index)
        
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context)
        return index
    except Exception as e:
        logging.error(f"Error in creating and persisting index: {e}")

# Query the engine
def query_engine(engine, query):
    try:
        response = engine.query(query)
        logging.info(f"Query response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error in querying engine: {e}")

# Display Markdown text
def display_markdown(text):
    try:
        display(Markdown(f"<b>{text}</b>"))
    except Exception as e:
        logging.error(f"Error in displaying markdown: {e}")

def main():
    try:
        query = input("Please provide your query! :--")
        setup_logging()
        logging.info("--------Logging Setup--------")
        service_context = initialize_llm_service()
        logging.info("--------LLM Setup--------")
        documents = load_and_preprocess_documents("Docs")
        logging.info("--------Docs loaded--------")
        index = create_and_persist_index(documents)
        logging.info("--------Index persisted--------")
        query_engine_instance = index.as_query_engine()
        logging.info("--------Query engine--------")
        response = query_engine(query_engine_instance, query)
        logging.info("--------Query fired--------")
        print("Query response:", response)
        display_markdown(response)
    except Exception as e:
        logging.error(f"Error in main function: {e}")

if __name__ == "__main__":
    main()
