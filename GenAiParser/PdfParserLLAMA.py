from llama_index.llms.openai import OpenAI
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)
from IPython.display import Markdown, display

documents = SimpleDirectoryReader("Docs").load_data()

print("Loaded documents:")
for doc in documents:
    doc.text = doc.text.upper()
    print(doc.text)

index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist()

'''
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context=storage_context)
query_engine = index.as_query_engine()
response = query_engine.query("What is the total experience?")
print(response)
display(Markdown(f"<b>{response}</b>"))
'''