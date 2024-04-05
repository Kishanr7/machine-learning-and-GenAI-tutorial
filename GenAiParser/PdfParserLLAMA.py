import logging
from datetime import datetime
from llama_index.llms.openai import OpenAI
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, StorageContext,
    load_index_from_storage, ServiceContext, set_global_service_context
)
from IPython.display import Markdown, display
import sys

# Setup logging
def setup_logging():
    try:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"../../GenAiParserLogs/log_{current_time}.log"
        logging.basicConfig(filename=log_filename,
                            level=logging.DEBUG,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
    except Exception as e:
        print(f"Error occurred while setting up logging: {str(e)}")
        sys.exit(1)

# Initialize LLM Service
def initialize_llm_service():
    try:
        llm = OpenAI(model='gpt-3.5-turbo', temperature=0, max_tokens=256)
        service_context = ServiceContext.from_defaults(llm=llm, chunk_size=1000, chunk_overlap=20)
        set_global_service_context(service_context)
        return service_context
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
def create_and_persist_index(documents, service_context):
    try:
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        index.storage_context.persist()
        return index
    except Exception as e:
        logging.error(f"Error occurred while creating and persisting index: {str(e)}")
        raise

# Load index from storage
def load_index(storage_dir):
    try:
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        return load_index_from_storage(storage_context=storage_context)
    except Exception as e:
        logging.error(f"Error occurred while loading index from storage: {str(e)}")
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
        service_context = initialize_llm_service()
        logging.info("--------LLM Setup--------")
        documents = load_and_preprocess_documents("Docs")
        logging.info("--------Docs loaded--------")
        index = create_and_persist_index(documents, service_context)
        logging.info("--------Index persisted--------")
        index = load_index("./storage")
        logging.info("--------Index stored--------")
        query_engine_instance = index.as_query_engine()
        logging.info("--------Query engine--------")
        response = query_engine(query_engine_instance, query)
        logging.info("--------Query fired--------")
        print("Query response:", response)
        display_markdown(response)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
