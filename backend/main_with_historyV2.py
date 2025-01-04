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

# Load environment variables
load_dotenv()

# Check required environment variables
required_env_vars = ["GOOGLE_API_KEY", "GROQ_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

# Initialize the LLM
llm = ChatGroq(model="gemma2-9b-it")

# Prompts for retrieval and question answering
retriever_prompt = (
    """Given a chat history and the latest user question which might reference context in the chat history,
    formulate a standalone question which can be understood without the chat history.
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", retriever_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

retriever = vector_store.as_retriever(k=3)
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

system_prompt = (
    """You are a helpful assistant. Answer questions with detailed and accurate information strictly based on the given context.
    Ensure your responses are concise, relevant, and do not include any references to external sources or unnecessary details.

    Context: {context}"""
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# FastAPI app initialization
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str
    chat_history: List[Dict[str, str]]

    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty')
        if len(v) > 1000:
            raise ValueError('Question must be less than 1000 characters')
        return v

@app.post("/ask")
async def ask_question(request: AskRequest):
    try:
        # Prepare chat history in the required format
        chat_history = [
            HumanMessage(message["content"]) if message["role"] == "human" else AIMessage(message["content"])
            for message in request.chat_history
        ]

        # Invoke the RAG chain
        response = rag_chain.invoke({
            "input": request.question,
            "chat_history": chat_history
        })

        # Extract retrieved documents and answer
        retrieved_docs = response["context"]
        answer = response["answer"]

        # Format retrieved document metadata
        docs_metadata = [
            {"file_name": doc.metadata.get("file_name", "unknown"), "page_content": doc.page_content}
            for doc in retrieved_docs
        ]

        return {"answer": answer, "retrieved_docs": docs_metadata}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Export the RAG chain for testing
__all__ = ["rag_chain"]
