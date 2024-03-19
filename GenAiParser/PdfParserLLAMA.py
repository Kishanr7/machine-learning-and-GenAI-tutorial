from llama_index.llms.openai import OpenAI
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    ServiceContext,
    set_global_service_context
)
from IPython.display import Markdown, display
import logging
import sys

logging.basicConfig(filename='1.log', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

llm = OpenAI(model ='gpt-3.5-turbo', temperature = 0, max_tokens = 256)

service_context = ServiceContext.from_defaults(llm=llm, chunk_size=1000, chunk_overlap= 20)
documents = SimpleDirectoryReader("Docs").load_data()

logging.info("Loaded documents:")
for doc in documents:
    doc.text = doc.text.upper()
    logging.info(doc.text)
    
index = VectorStoreIndex.from_documents(documents, service_context= service_context)
index.storage_context.persist()

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context=storage_context)
query_engine = index.as_query_engine()
response = query_engine.query("What is the total experience?")
print(response)
display(Markdown(f"<b>{response}</b>"))