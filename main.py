from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document


data=None
local_path = "WEF_The_Global_Cooperation_Barometer_2024.pdf"
print('hello')
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
print(data[0].page_content)

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
chunks = text_splitter.split_documents(documents)

# Print the chunks
print(chunks, 'hello last')