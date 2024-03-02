import os
import langchain
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Pinecone
from langchain_community.vectorstores import Pinecone
# import pinecone
# from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.prompts import PromptTemplate
# from langchain.llms import CTransformers
from langchain_community.llms import CTransformers
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv


load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

#Extract data from the PDF
def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    
    documents = loader.load()

    return documents
extracted_data = load_pdf("data/")
# print(extracted_data[0])

#Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks

text_chunks = text_split(extracted_data)
print("length of my chunk:", len(text_chunks))

# #download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
embeddings = download_hugging_face_embeddings()
print(embeddings)

query_result = embeddings.embed_query("Hello world")
print("Length", len(query_result))

# print(text_chunks[0].page_content)
# print([t.page_content for t in text_chunks])

index_name="testing"

#Creating Embeddings for Each of The Text Chunks & storing
# docsearch=Pinecone([t.page_content for t in text_chunks], embeddings, index_name=index_name)


# index_name = "langchain-test-index"

docsearch = PineconeVectorStore.from_documents([t.page_content for t in text_chunks], embeddings, index_name=index_name)


# #If we already have an index we can load it like this
# # docsearch=Pinecone.from_existing_index(index_name, embeddings)

query = "What are Allergies"

docs=docsearch.similarity_search(query, k=3)

print("Result", docs)

