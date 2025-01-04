import os
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import (
    HumanMessage,
    trim_messages,
)


from dotenv import load_dotenv

load_dotenv()



word_docs_folder = 'word_docs'

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

llm = ChatGroq(model="llama-3.3-70b-versatile")

trimmer = trim_messages(

    strategy="last",
    token_counter=len,
    max_tokens=7,
    # Usually, we want to keep the SystemMessage
    # if it's present in the original history.
    # The SystemMessage has special instructions for the model.
    include_system=True,
    # Most chat models expect that chat history starts with either:
    # (1) a HumanMessage or
    # (2) a SystemMessage followed by a HumanMessage
    # start_on="human" makes sure we produce a valid chat history
    start_on="human",
)

llm_chain = trimmer | llm


vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

# for file_name in os.listdir(word_docs_folder):
#     print(f"Processing file: {file_name}")
#     file_path = os.path.join(word_docs_folder, file_name)

#     doc = Document(file_path)
#     full_texts = []
#     for para in doc.paragraphs:
#         full_texts.append(para.text)

#     content = '\n'.join(full_texts)
#     contents = [content]
        
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

#     metadata = {
#         "file_name": file_name,
#         "file_path": file_path
#     }

#     metadatas = [metadata]

#     text_chunks = text_splitter.create_documents(texts=contents, metadatas=metadatas)

#     vector_store.add_documents(documents=text_chunks)
#     print(f"Documents of file {file_name} added to the vector store")


retriever_prompt = (
    """Given a chat history and the latest user question which might reference context in the chat history,
    formulate a standalone question which can be understood without the chat history.
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
)
    
contextualize_q_prompt  = ChatPromptTemplate.from_messages(
    [
        ("system", retriever_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

retriever = vector_store.as_retriever(k=3)

history_aware_retriever = create_history_aware_retriever(llm_chain ,retriever, contextualize_q_prompt)

system_prompt = (
    """You are a helpful assistant. Answer questions with detailed and accurate information strictly based on the given context.
    Ensure your responses are concise, relevant, and do not include any references to external sources or unnecessary details.
    Avoid introductory phrases like "Based on the context provided

    Context:{context}"""
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm_chain, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)




store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    # print(store[session_id])
    print("-----------------------------------")


    print(len(store[session_id].messages))

    
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


while True:
    question = input("Ask a question: ")

    response = conversational_rag_chain.invoke(
        {"input": question},
        config={"session_id": "2"}
    )

    # print(response)

    retrieved_docs = response["context"]
    
    answer = response["answer"]

    print(f"Answer: {answer}")

    print("-----------------------------------")

    # for result in retrieved_docs:
    #     print(f"Document: {result.metadata['file_name']}")
    #     print(f"Page content: {result.page_content}")
    #     print("-----------------------------------")

    # retrieved_docs = retriever.invoke(question)

    # context = ""
    # for result in retrieved_docs:
    #     context += result.page_content + "\n"

    # print(f"Context: {context}")

    # print("-----------------------------------")

    # prompt = f"""Answer the question in detail based on the following context only: 
    # {context} 
    # Guidelines for answering:
    # 1. Do not refer to any external sources.
    # 2. Do not provide any irrelevant information.
    # 3. No need to start with text like 'based on the context provided' or 'according to the context'.

    # Question: {question}"""

    # answer = llm.invoke(prompt)

    # answer_text = answer.content
    # print(f"Answer: {answer_text}")

    # print("-----------------------------------")

    # for result in retrieved_docs:
    #     print(f"Document: {result.metadata['file_name']}")
    #     print(f"Page content: {result.page_content}")
    #     print("-----------------------------------")
