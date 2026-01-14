import os
import threading
import time
import json
import streamlit as st
import whisper
import chromadb
from chromadb.utils import embedding_functions
import torch
import base64

# --- Configuration ---
BASE_DB_FOLDER = "Database"
PROCESSING_FOLDER = os.path.join(BASE_DB_FOLDER, "processing")  # תיקייה למעקב
device = "cuda" if torch.cuda.is_available() else "cpu"

# וודא שהתיקייה קיימת
if not os.path.exists(PROCESSING_FOLDER):
    os.makedirs(PROCESSING_FOLDER)


# --- Helper: User Paths ---
def get_user_paths(username):
    user_folder = os.path.join(BASE_DB_FOLDER, "users", username)
    videos_dir = os.path.join(user_folder, "videos")
    chroma_dir = os.path.join(user_folder, "chroma_db")

    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(chroma_dir, exist_ok=True)

    return videos_dir, chroma_dir


def get_safe_collection_name(video_name):
    safe_hash = base64.b64encode(video_name.encode()).decode().replace("=", "").replace("/", "_").replace("+", "-")
    return f"vid_{safe_hash}"


# --- Status Management Functions ---
def update_progress(username, video_name, progress_percent, stage_name):
    """Writes the current status to a JSON file."""
    safe_name = get_safe_collection_name(video_name)
    status_file = os.path.join(PROCESSING_FOLDER, f"{username}_{safe_name}.json")

    # Create dictionary with status data
    status_data = {
        "video": video_name,
        "progress": progress_percent,  # Integer 0-100
        "stage": stage_name  # Text like "Transcribing..."
    }

    with open(status_file, "w") as f:
        json.dump(status_data, f)


def clear_progress(username, video_name):
    """Deletes the status file when finished."""
    safe_name = get_safe_collection_name(video_name)
    status_file = os.path.join(PROCESSING_FOLDER, f"{username}_{safe_name}.json")
    if os.path.exists(status_file):
        os.remove(status_file)


def get_active_progress(username):
    """Reads all status files for this user."""
    active_jobs = []
    # Find all .json files starting with username_
    for f in os.listdir(PROCESSING_FOLDER):
        if f.startswith(f"{username}_") and f.endswith(".json"):
            try:
                with open(os.path.join(PROCESSING_FOLDER, f), "r") as file:
                    data = json.load(file)
                    active_jobs.append(data)
            except:
                continue
    return active_jobs


# --- Backend Logic ---
@st.cache_resource
def load_whisper():
    return whisper.load_model("small", device=device)


def get_db_client(chroma_path):
    return chromadb.PersistentClient(path=chroma_path)


def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def delete_video(username, video_name):
    videos_dir, chroma_dir = get_user_paths(username)
    client = get_db_client(chroma_dir)
    col_name = get_safe_collection_name(video_name)
    try:
        client.delete_collection(col_name)
    except:
        pass
    file_path = os.path.join(videos_dir, video_name)
    if os.path.exists(file_path):
        os.remove(file_path)
    clear_progress(username, video_name)
    return True


def process_video_in_background(file_path, video_name, chroma_path, username):
    # 1. Start - 0%
    update_progress(username, video_name, 5, "Extracting Audio & Transcribing...")

    try:
        model = load_whisper()
        client = get_db_client(chroma_path)
        ef = get_embedding_function()
        collection_name = get_safe_collection_name(video_name)

        # Cleanup old collection if exists
        try:
            client.delete_collection(collection_name)
        except:
            pass

        collection = client.create_collection(name=collection_name, embedding_function=ef)

        # 2. Transcription (This takes the longest time)
        # Note: Whisper blocks the thread, so it will sit at 10% until finished.
        update_progress(username, video_name, 10, "Transcribing Audio (This may take a while)...")

        result = model.transcribe(file_path)
        segments = result['segments']

        # 3. Embedding Loop - We can calculate exact progress here
        GROUP_SIZE = 3
        total_groups = len(segments) // GROUP_SIZE + 1

        ids = []
        documents = []
        metadatas = []

        for idx, i in enumerate(range(0, len(segments), GROUP_SIZE)):
            group = segments[i: i + GROUP_SIZE]
            combined_text = " ".join([s['text'].strip() for s in group])

            if not group: continue

            ids.append(f"{collection_name}_{i}")
            documents.append(combined_text)
            metadatas.append({
                "start_time": group[0]['start'],
                "end_time": group[-1]['end'],
                "video_name": video_name,
                "source_collection": collection_name
            })

            # Add to DB immediately or in batch? We'll do batch for speed,
            # but update progress calculation:
            # Progress maps from 40% (post-transcribe) to 90% (finished embedding)
            current_progress = 40 + int((idx / total_groups) * 50)
            update_progress(username, video_name, current_progress, "Indexing Knowledge...")

        collection.add(ids=ids, documents=documents, metadatas=metadatas)

        # 4. Finish
        update_progress(username, video_name, 100, "Done!")
        time.sleep(2)  # Let the user see 100% for a moment

    except Exception as e:
        print(f"Error processing video: {e}")
        update_progress(username, video_name, 0, f"Error: {str(e)[:20]}")

    finally:
        clear_progress(username, video_name)


# --- UI Functions ---

def get_videos_list(username):
    videos_dir, _ = get_user_paths(username)
    if not os.path.exists(videos_dir):
        return []
    return [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.mov', '.avi'))]


def render_upload_page(username):
    st.header("☁️ Upload Center")
    st.caption(f"Storage for: {username}")
    videos_dir, chroma_dir = get_user_paths(username)

    with st.container(border=True):
        uploaded_file = st.file_uploader("Drag and drop video here", type=["mp4", "mov", "avi"])
        if uploaded_file:
            file_path = os.path.join(videos_dir, uploaded_file.name)
            if st.button("Start Processing", type="primary"):
                # Use chunked writing to save RAM
                with open(file_path, "wb") as f:
                    while True:
                        chunk = uploaded_file.read(5 * 1024 * 1024)
                        if not chunk: break
                        f.write(chunk)

                st.toast(f"File saved! Starting background processor...")

                thread = threading.Thread(
                    target=process_video_in_background,
                    args=(file_path, uploaded_file.name, chroma_dir, username)
                )
                thread.start()
                time.sleep(1)  # Wait a sec for the file to be created
                st.rerun()


def render_library_page(username):
    st.header(f"📚 {username}'s Library")

    # Check for active processing jobs
    active_jobs = get_active_progress(username)
    if active_jobs:
        with st.status("🔄 Processing New Videos...", expanded=True):
            for job in active_jobs:
                st.write(f"**{job['video']}**: {job['stage']}")
                st.progress(job['progress'])

    videos = get_videos_list(username)
    if not videos:
        st.info("Your library is empty.")
        return

    search = st.text_input("Filter library...", "")
    filtered_videos = [v for v in videos if search.lower() in v.lower()]

    for vid in filtered_videos:
        with st.container(border=True):
            col_text, col_open, col_del = st.columns([5, 1.5, 0.5])
            with col_text:
                st.subheader(f"🎬 {vid}")
            with col_open:
                st.write("")
                if st.button("Open Workspace", key=f"btn_{vid}", use_container_width=True):
                    st.session_state['selected_video'] = vid
                    st.session_state['current_page'] = "Chat Workspace"
                    st.rerun()
            with col_del:
                st.write("")
                if st.button("🗑️", key=f"del_{vid}"):
                    if delete_video(username, vid):
                        st.success(f"Deleted {vid}")
                        st.rerun()