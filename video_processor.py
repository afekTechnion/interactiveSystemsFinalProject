import os
import threading
import streamlit as st
import whisper
import chromadb
from chromadb.utils import embedding_functions
import torch

# --- Configuration ---
BASE_DB_FOLDER = "Database"
VIDEOS_DIR = os.path.join(BASE_DB_FOLDER, "videos_db")
CHROMA_DB_DIR = os.path.join(BASE_DB_FOLDER, "transcription_db")

os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


# --- Backend Logic ---
@st.cache_resource
def load_whisper():
    return whisper.load_model("small", device=device)


@st.cache_resource
def get_db_client():
    return chromadb.PersistentClient(path=CHROMA_DB_DIR)


def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def process_video_in_background(file_path, video_name):
    # (הקוד כאן נשאר זהה למה שהיה לך מקודם - ללא שינוי)
    try:
        model = load_whisper()
        client = get_db_client()
        ef = get_embedding_function()
        collection_name = "".join([c if c.isalnum() else "_" for c in video_name])
        try:
            client.delete_collection(collection_name)
        except:
            pass
        collection = client.create_collection(name=collection_name, embedding_function=ef)
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


# --- UI Functions (החלק החדש) ---

def get_videos_list():
    """Returns a list of video filenames."""
    if not os.path.exists(VIDEOS_DIR):
        return []
    return [f for f in os.listdir(VIDEOS_DIR) if f.endswith(('.mp4', '.mov', '.avi'))]


def render_upload_page():
    """מציג את מסך ההעלאה בלבד"""
    st.header("☁️ Upload Center")
    st.write("Upload new videos to your knowledge base.")

    # אזור גרירה מעוצב
    with st.container(border=True):
        uploaded_file = st.file_uploader("Drag and drop video here", type=["mp4", "mov", "avi"])

        if uploaded_file:
            file_path = os.path.join(VIDEOS_DIR, uploaded_file.name)

            # כפתור שמפעיל את התהליך
            if st.button("Start Processing", type="primary"):
                if not os.path.exists(file_path):
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    st.success(f"File saved: {uploaded_file.name}")

                    # הרצת תהליך ברקע
                    thread = threading.Thread(target=process_video_in_background, args=(file_path, uploaded_file.name))
                    thread.start()
                    st.info("Processing started in background! You can go to the Library now.")
                else:
                    st.warning("File already exists.")


def render_library_page():
    """מציג את כל הסרטונים כרשימה/גריד"""
    st.header("📚 Video Library")

    videos = get_videos_list()

    if not videos:
        st.info("No videos found. Go to 'Upload' to add some!")
        return

    # חיפוש בתוך הספרייה (פילטר פשוט)
    search = st.text_input("Filter library...", "")
    filtered_videos = [v for v in videos if search.lower() in v.lower()]

    # הצגת הסרטונים
    for vid in filtered_videos:
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.subheader(f"🎬 {vid}")
            with col2:
                # כפתור שמעביר למסך העבודה (Chat)
                if st.button("Open Workspace", key=f"btn_{vid}"):
                    st.session_state['selected_video'] = vid
                    st.session_state['current_page'] = "Chat Workspace"  # ניווט אוטומטי
                    st.rerun()