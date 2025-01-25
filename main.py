from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

data=None
# local_path = "WEF_The_Global_Cooperation_Barometer_2024.pdf"
local_path = "scammer-agent.pdf"
# print('hello')
#
# # Local PDF file uploads
if local_path:
    loader = UnstructuredPDFLoader(file_path=local_path)
    data = loader.load()
else:
    print("Upload a PDF file")

# print('hello after')
#
# # Preview first page
# data[0].page_content
# print(data[0].page_content)

print('====================')
print('====================')
print('====================')
print('====================')

#########

#
# # Split and chunk
# Create documents as LangChain's Document objects
documents = [
    Document(page_content="Hello", metadata={}),
    Document(page_content="world", metadata={}),
]

# Define the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)

# Split the documents
chunks = text_splitter.split_documents(data)

# Add to vector database
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
    collection_name="local-rag"
)

## RETRIEVAL
# LLM from Ollama
local_model = "llama3.2:1b"  # Replace with your model name
llm = ChatOllama(model=local_model)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines.
Original question: {question}"""
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(),
    llm,
    prompt=QUERY_PROMPT
)

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# chain.invoke(input="")
response = chain.invoke(input="Can you explain the case study highlighted in the document?")
print(response)

# Print the chunks
# print(chunks, 'hello last')