# src/retriever/retriever.py
import os
import glob
from typing import List

# --- CLEANED UP IMPORTS ---
# This is the single, correct import for Chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
# This is the single, correct import for HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

# --- Configuration ---
DATA_PATH = "data/corpus"
DB_PATH = "chroma_db/"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Main Retriever Creation Function ---

def create_retriever(k_results: int = 3) -> VectorStoreRetriever:
    """
    Creates a ChromaDB vector store and returns a retriever.
    """
    # Initialize the embedding model
    print(f"Initializing embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )

    if os.path.exists(DB_PATH):
        # Load the existing database
        print(f"Loading existing vector store from: {DB_PATH}")
        vectorstore = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings
        )
    else:
        # Create a new database
        print("Existing vector store not found. Creating a new one...")
        print(f"Loading documents from: {DATA_PATH}")
        documents = load_documents()
        if not documents:
            raise ValueError("No documents found. Please add text files to the data/corpus directory.")

        print(f"Splitting {len(documents)} documents into chunks...")
        chunks = split_documents(documents)
        print(f"Split into {len(chunks)} chunks.")

        print("Creating new vector store and embedding documents...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_PATH
        )
        print("Vector store created successfully.")

    # Create and return the retriever
    print(f"Creating retriever with k={k_results}")
    return vectorstore.as_retriever(search_kwargs={'k': k_results})

def load_documents() -> List[Document]:
    """Loads all .txt files from the data path."""
    document_list = []
    for filepath in glob.glob(os.path.join(DATA_PATH, "*.txt")):
        loader = TextLoader(filepath, encoding='utf-8')
        document_list.extend(loader.load())
    return document_list

def split_documents(documents: List[Document]) -> List[Document]:
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


# --- Self-testing Block ---
if __name__ == "__main__":
    print("--- Running Retriever Self-Test ---")
    if not os.path.exists(os.path.join(DATA_PATH, "test_doc.txt")):
         with open(os.path.join(DATA_PATH, "test_doc.txt"), "w") as f:
            f.write("LangGraph is a library for building stateful, multi-actor applications with LLMs.\n")

    retriever = create_retriever()
    print(f"\nRetriever created successfully: {type(retriever)}")

    sample_query = "What is LangGraph?"
    print(f"\n--- Testing retrieval with query: '{sample_query}' ---")
    try:
        retrieved_docs = retriever.invoke(sample_query)
        print(f"Retrieved {len(retrieved_docs)} documents.")
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs):
                print(f"\n--- Document {i+1} ---\nContent: {doc.page_content}\nSource: {doc.metadata.get('source', 'N/A')}")
    except Exception as e:
        print(f"An error occurred during retrieval test: {e}")

    print("\n--- Retriever Self-Test Complete ---")