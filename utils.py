import json
import os
import streamlit as st
from pathlib import Path
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from huggingface_hub import InferenceClient

def load_data(data_path: str) -> Tuple[List[str], List[Dict]]:
    """
    Loads clinical notes from JSON files in the specified directory.
    Parses input sections and combines them into a single text document.
    """
    documents = []
    metadatas = []
    
    # Walk through the directory
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # Combine input sections (input1 to input6)
                    full_text = ""
                    for i in range(1, 7):
                        key = f"input{i}"
                        if key in data:
                            full_text += f"Section {i}: {data[key]}\n"
                    
                    if full_text.strip():
                        documents.append(full_text)
                        
                        # Extract diagnosis from path or filename as metadata
                        path_parts = Path(file_path).parts
                        diagnosis = "Unknown"
                        if len(path_parts) >= 3:
                            diagnosis = f"{path_parts[-3]} - {path_parts[-2]}"
                            
                        metadatas.append({
                            "source": file,
                            "diagnosis": diagnosis,
                            "path": file_path
                        })
                except Exception:
                    continue

    return documents, metadatas

def preprocess_text(text: str) -> List[str]:
    """
    Tokenizes text, removing punctuation and common stop words.
    """
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", 
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "it", "this", "that", "these", "those", "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
        "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
        "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "can", "will", "just", "should", "now"
    }
    
    text = text.lower()
    for char in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~':
        text = text.replace(char, ' ')
        
    tokens = text.split()
    return [t for t in tokens if t not in stop_words]

def create_retriever(documents: List[str]):
    """
    Creates a BM25 retriever from the list of documents.
    """
    tokenized_corpus = [preprocess_text(doc) for doc in documents]
    return BM25Okapi(tokenized_corpus)

def load_llm():
    """
    Initializes the Hugging Face Inference Client using an API token.
    Uses st.secrets for secure token management on Streamlit Cloud.
    """
    # Get token from secrets
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except Exception:
        st.error("HF_TOKEN not found in Streamlit Secrets. Please add it to deploy.")
        return None

    # Using TinyLlama Chat model via Inference API
    client = InferenceClient(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        token=hf_token
    )
    return client

def rag_pipeline(query: str, bm25, documents: List[str], metadatas: List[Dict], client, k: int = 3):
    """
    Runs the RAG pipeline using the HF Inference API.
    """
    if client is None:
        return "Error: LLM Client not initialized. Please check your API token.", []

    # 1. Retrieve
    tokenized_query = preprocess_text(query)
    scores = bm25.get_scores(tokenized_query)
    top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    
    retrieved_docs = [documents[i] for i in top_n_indices]
    retrieved_metas = [metadatas[i] for i in top_n_indices]
    
    # 2. Prepare Context
    context = ""
    for i, doc in enumerate(retrieved_docs):
        # Snippet to keep prompt size manageable
        context += f"Document {i+1} (Diagnosis: {retrieved_metas[i]['diagnosis']}):\n{doc[:1000]}...\n\n"
        
    # 3. Generate using Chat format
    messages = [
        {"role": "system", "content": "You are a medical assistant. Answer the user's question strictly based on the provided context. If the answer is not in the context, say 'The provided documents do not contain the answer.'"},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    
    try:
        response = ""
        for message in client.chat_completion(
            messages,
            max_tokens=512,
            temperature=0.1,
            stream=True
        ):
            token = message.choices[0].delta.content
            if token:
                response += token
        
        return response.strip(), retrieved_metas
    except Exception as e:
        return f"API Error: {str(e)}", []
