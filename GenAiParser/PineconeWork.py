import logging
from datetime import datetime
import os
import sys
from IPython.display import Markdown, display
from llama_index.llms.openai import OpenAI
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, StorageContext,
    Settings, ServiceContext
)
import time
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, PodSpec

# Setup logging
def setup_logging():
    try:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"../../GenAiParserLogs/log_PC_w_{current_time}.log"
        logging.basicConfig(filename=log_filename,
                            level=logging.DEBUG,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    except Exception as e:
        print(f"Error occurred while setting up logging: {str(e)}")
        sys.exit(1)
        
# Initialize LLM Service
def initialize_llm_service():
    try:
        Settings.llm = OpenAI(model='gpt-3.5-turbo', temperature=0, max_tokens=256)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
        Settings.chunk_size = 1000
        Settings.chunk_overlap = 20
        return Settings
    except Exception as e:
        logging.error(f"Error occurred while initializing LLM service: {str(e)}")
        raise


# Load and preprocess documents
def load_and_preprocess_documents(path):
    try:
        documents = SimpleDirectoryReader(path).load_data()
        logging.info("Documents loaded and preprocessed.")
        return documents
    except Exception as e:
        logging.error(f"Error occurred while loading and preprocessing documents: {str(e)}")
        raise

# Create and persist index
def create_and_persist_index(documents, Settings):
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
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)
            
        pinecone_index = pc.Index(index_name)
        print(pinecone_index.describe_index_stats())
        vector_store =  PineconeVectorStore(pinecone_index=pinecone_index)
        
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context)
        logging.info("Index persisted in Pinecone.")
        return index

    except Exception as e:
        logging.error(f"Error occurred while creating and persisting index: {str(e)}")
        raise

# Query the engine
def query_engine(engine, query):
    try:
        response = engine.query(query)
        logging.info(f"Query response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error occurred while querying the engine: {str(e)}")
        raise

# Display Markdown text
def display_markdown(text):
    display(Markdown(f"<b>{text}</b>"))

def main():
    try:
        query = input("Please provide your query! :--")
        setup_logging()
        logging.info("--------Logging Setup--------")
        settings = initialize_llm_service()
        logging.info("--------LLM Setup--------")
        documents = load_and_preprocess_documents("Docs")
        logging.info("--------Docs loaded--------")
        for i in range(len(documents)-1):
            logging.info(f'{i} document: {documents[i].text}')
        '''index = create_and_persist_index(documents, settings)
        logging.info("--------Index persisted--------")
        query_engine_instance = index.as_query_engine()
        logging.info("--------Query engine--------")
        response = query_engine(query_engine_instance, query)
        logging.info("--------Query fired--------")
        print("Query response:", response)
        display_markdown(response)'''
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()