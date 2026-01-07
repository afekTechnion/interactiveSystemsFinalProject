import os
import threading
import streamlit as st
import whisper
import chromadb
from chromadb.utils import embedding_functions
import torch

# Configuration
BASE_DB_FOLDER = "Database"
VIDEOS_DIR = os.path.join(BASE_DB_FOLDER, "videos_db")
CHROMA_DB_DIR = os.path.join(BASE_DB_FOLDER, "transcription_db")

# Ensure directories exist
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_whisper():
    return whisper.load_model("small", device=device)


@st.cache_resource
def get_db_client():
    return chromadb.PersistentClient(path=CHROMA_DB_DIR)


def get_embedding_function():
    # Using the specific model from your code
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def process_video_in_background(file_path, video_name):
    try:
        model = load_whisper()
        client = get_db_client()
        ef = get_embedding_function()

        # Create valid collection name
        collection_name = "".join([c if c.isalnum() else "_" for c in video_name])

        # Reset collection if exists to avoid duplicates
        try:
            client.delete_collection(collection_name)
        except:
            pass

        collection = client.create_collection(name=collection_name, embedding_function=ef)

        # Transcribe
        result = model.transcribe(file_path)

        ids = []
        documents = []
        metadatas = []

        for i, segment in enumerate(result['segments']):
            text = segment['text'].strip()
            ids.append(f"{collection_name}_{i}")
            documents.append(text)
            metadatas.append({
                "start_time": segment['start'],
                "end_time": segment['end'],
                "video_name": video_name,
                "source_collection": collection_name
            })

        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        print(f"Finished processing {video_name}")

    except Exception as e:
        print(f"Error processing video: {e}")


def render_sidebar_ui():
    """Renders the upload and library sidebar."""
    st.sidebar.header("Library & Upload")

    # Initialize session state for videos if not present
    if 'processed_videos' not in st.session_state:
        st.session_state['processed_videos'] = os.listdir(VIDEOS_DIR)

    uploaded_file = st.sidebar.file_uploader("Upload New Video", type=["mp4", "mov"])

    if uploaded_file:
        file_path = os.path.join(VIDEOS_DIR, uploaded_file.name)

        # Avoid re-uploading/processing if exists
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.toast(f"Saved {uploaded_file.name}")

            # Start Background Thread
            thread = threading.Thread(target=process_video_in_background, args=(file_path, uploaded_file.name))
            thread.start()
            st.sidebar.info("Processing started in background...")

            # Update list
            st.session_state['processed_videos'].append(uploaded_file.name)

    st.sidebar.divider()
    st.sidebar.subheader("Your Videos")

    video_list = st.session_state['processed_videos']
    if video_list:
        selected = st.sidebar.radio("Select video:", video_list)
        return selected
    else:
        st.sidebar.info("No videos yet.")
        return None
