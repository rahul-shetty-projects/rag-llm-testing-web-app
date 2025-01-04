from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List, Dict
from langchain_core.messages.chat import ChatMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Check if all required environment variables are set
required_env_vars = ["GOOGLE_API_KEY", "GROQ_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize the embedding model for vectorization of documents and queries
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize Chroma as the vector database for storing and retrieving embeddings
vector_store = Chroma(
    collection_name="example_collection",  # Name of the collection in the vector store
    embedding_function=embeddings,         # Function to generate embeddings
    persist_directory="./chroma_langchain_db",  # Directory to persist the vector store
)

# Initialize the language model (LLM) for generating responses
llm = ChatGroq(model="gemma2-9b-it")

# Define a system prompt for retriever chain to rephrase questions based on chat history
retriever_prompt = (
    """Given a chat history and the latest user question which might reference context in the chat history,
    formulate a standalone question which can be understood without the chat history.
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
)

# Template for chat-based prompting in retrieval process
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", retriever_prompt),
    MessagesPlaceholder(variable_name="chat_history"),  # Placeholder for chat history
    ("human", "{input}"),  # Latest user question as input
])

# Set up the retriever to fetch relevant documents from the vector store
retriever = vector_store.as_retriever(k=3)  # Retrieve top 3 most relevant documents
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# Define a system prompt for the QA chain to answer questions based on context
system_prompt = (
    """You are a helpful assistant. Answer questions with detailed and accurate information strictly based on the given context.
    Ensure your responses are concise, relevant, and do not include any references to external sources or unnecessary details.

    Context: {context}"""
)

# Template for chat-based prompting in the question-answering process
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),  # System's instructions
    MessagesPlaceholder("chat_history"),  # Placeholder for chat history
    ("human", "{input}"),  # User's input/question
])

# Create a question-answering chain that combines context and user queries
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval-augmented generation (RAG) chain combining retrieval and QA chains
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Define a request schema for the `/ask` endpoint
class AskRequest(BaseModel):
    question: str  # User's question
    chat_history: List[Dict[str, str]]  # List of previous chat messages

    @validator('question')
    def validate_question(cls, v):
        # Validate the question field
        if not v.strip():
            raise ValueError('Question cannot be empty')
        if len(v) > 1000:
            raise ValueError('Question must be less than 1000 characters')
        return v

# Define the `/ask` endpoint for asking questions
@app.post("/ask")
async def ask_question(request: AskRequest):
    try:
        # Prepare the chat history in the format required by LangChain
        chat_history = [
            HumanMessage(message["content"]) if message["role"] == "human" else AIMessage(message["content"])
            for message in request.chat_history
        ]

        # Invoke the RAG chain to generate an answer based on the question and chat history
        response = rag_chain.invoke({
            "input": request.question,
            "chat_history": chat_history
        })

        # Extract retrieved documents and the generated answer
        retrieved_docs = response["context"]
        answer = response["answer"]

        # Format retrieved documents' metadata
        docs_metadata = [
            {"file_name": doc.metadata.get("file_name", "unknown"), "page_content": doc.page_content}
            for doc in retrieved_docs
        ]

        # Return the answer and metadata of retrieved documents
        return {"answer": answer, "retrieved_docs": docs_metadata}

    except Exception as e:
        # Handle exceptions and return an internal server error
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Export the RAG chain for external testing or integration
__all__ = ["rag_chain"]
