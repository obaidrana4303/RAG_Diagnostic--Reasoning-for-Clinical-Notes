import streamlit as st
import os
from utils import load_data, create_retriever, load_llm, rag_pipeline

# Page Config
st.set_page_config(page_title="DiReCT: Diagnostic Reasoning", layout="wide")

st.title("Clinical Diagnostic Reasoning Assistant (Cloud Optimized)")
st.markdown("Ask questions about clinical notes. This version uses the Hugging Face Inference API to stay within Streamlit's memory limits.")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    # Default path based on user's environment
    default_path = r"mimic-iv-ext-direct-1.0.0/Finished"
    data_path = st.text_input("Data Directory Path", value=default_path)
    
    if st.button("Reload Data"):
        st.cache_resource.clear()
        st.success("Cache cleared. Reloading...")

# Load Resources (Cached)
@st.cache_resource
def get_resources(path):
    with st.spinner("Loading clinical notes..."):
        documents, metadatas = load_data(path)
    
    if not documents:
        return None, None, None, None

    with st.spinner("Building retriever..."):
        bm25 = create_retriever(documents)
        
    with st.spinner("Connecting to Hugging Face API..."):
        llm = load_llm()
        
    return documents, metadatas, bm25, llm

# Main App Logic
if os.path.exists(data_path):
    try:
        documents, metadatas, bm25, llm = get_resources(data_path)
        
        if documents is None:
            st.warning(f"No clinical notes found in: {data_path}. Please check the path.")
        else:
            st.success(f"Successfully indexed {len(documents)} clinical notes.")
            
            if llm is None:
                st.error("Hugging Face Client failed to initialize. Did you add `HF_TOKEN` to Streamlit Secrets?")
            else:
                # User Input
                query = st.text_area("Enter your clinical question:", height=100, placeholder="e.g., What are the symptoms of the patient with suspected epilepsy?")
                
                if st.button("Analyze"):
                    if query:
                        with st.spinner("Querying API and generating response..."):
                            response, retrieved_metas = rag_pipeline(query, bm25, documents, metadatas, llm)
                        
                        # Display Answer
                        st.subheader("Diagnostic Reasoning:")
                        st.markdown(response)
                        
                        # Display Sources
                        if retrieved_metas:
                            st.subheader("Retrieved Context Sources:")
                            for i, meta in enumerate(retrieved_metas):
                                with st.expander(f"Source {i+1}: {meta['diagnosis']}"):
                                    st.write(f"File: {meta['source']}")
                                    st.write(f"Path: {meta['path']}")
                    else:
                        st.warning("Please enter a question.")
                
    except Exception as e:
        st.error(f"App Error: {str(e)}")
else:
    st.error(f"Data directory not found: {data_path}")