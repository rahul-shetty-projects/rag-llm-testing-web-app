import os
import shutil
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# Delete old Chroma database if it exists (since we're using a different embedding model)
chroma_db_path = "./chroma_langchain_db"
if os.path.exists(chroma_db_path):
    print("Removing old Chroma database")
    shutil.rmtree(chroma_db_path)

# Initialize embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

# Folder containing Word documents
word_docs_folder = "word_docs"

def process_and_store_documents():
    for file_name in os.listdir(word_docs_folder):
        if not file_name.endswith(".docx"):
            continue
        
        print(f"Processing file: {file_name}")
        file_path = os.path.join(word_docs_folder, file_name)

        # Read document content
        doc = Document(file_path)
        full_texts = [para.text for para in doc.paragraphs]
        content = '\n'.join(full_texts)
        contents = [content]

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        metadata = {"file_name": file_name, "file_path": file_path}
        metadatas = [metadata]
        text_chunks = text_splitter.create_documents(texts=contents, metadatas=metadatas)

        # Add chunks to vector store
        vector_store.add_documents(documents=text_chunks)
        print(f"Documents of file {file_name} added to the vector store")


if __name__ == "__main__":
    process_and_store_documents()
