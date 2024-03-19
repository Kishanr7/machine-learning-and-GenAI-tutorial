from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from IPython.display import Markdown, display

documents = SimpleDirectoryReader("Docs").load_data()

print("Loaded documents:")
for doc in documents:
    doc.text = doc.text.upper()
    print(doc.text)

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What is the total experience?")
print(response)
display(Markdown(f"<b>{response}</b>"))
