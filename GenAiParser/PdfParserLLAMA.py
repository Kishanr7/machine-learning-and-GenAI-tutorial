from llama_index.llms.openai import OpenAI
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, StorageContext,
    load_index_from_storage, ServiceContext, set_global_service_context
)
from IPython.display import Markdown, display
import logging
import sys

# Setup logging
def setup_logging():
    logging.basicConfig(filename='1.log', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Initialize LLM Service
def initialize_llm_service():
    llm = OpenAI(model='gpt-3.5-turbo', temperature=0, max_tokens=256)
    service_context = ServiceContext.from_defaults(llm=llm, chunk_size=1000, chunk_overlap=20)
    set_global_service_context(service_context)
    return service_context

# Load and preprocess documents
def load_and_preprocess_documents(path):
    documents = SimpleDirectoryReader(path).load_data()
    for doc in documents:
        doc.text = doc.text.upper()
    logging.info("Documents loaded and preprocessed.")
    return documents

# Create and persist index
def create_and_persist_index(documents, service_context):
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    index.storage_context.persist()
    return index

# Load index from storage
def load_index(storage_dir):
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    return load_index_from_storage(storage_context=storage_context)

# Query the engine
def query_engine(engine, query):
    response = engine.query(query)
    logging.info(f"Query response: {response}")
    return response

# Display Markdown text
def display_markdown(text):
    display(Markdown(f"<b>{text}</b>"))

def main():
    setup_logging()
    print("--------Logging Setup--------")
    service_context = initialize_llm_service()
    print("--------LLM Setup--------")
    documents = load_and_preprocess_documents("Docs")
    print("--------Docs loaded--------")
    index = create_and_persist_index(documents, service_context)
    print("--------index persisted--------")
    index = load_index("./storage")
    print("--------index stored--------")
    query_engine_instance = index.as_query_engine()
    print("--------Query engine--------")
    response = query_engine(query_engine_instance, "What are the key competencies of the author?")
    print("--------Query fired--------")
    print(response)
    display_markdown(response)

if __name__ == "__main__":
    main()