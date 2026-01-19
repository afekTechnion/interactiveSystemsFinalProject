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
import cv2  # Needed for thumbnails

# --- Configuration ---
BASE_DB_FOLDER = "Database"
PROCESSING_FOLDER = os.path.join(BASE_DB_FOLDER, "processing")
device = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(PROCESSING_FOLDER):
    os.makedirs(PROCESSING_FOLDER)


# --- Helper: User Paths ---
def get_user_paths(username):
    user_folder = os.path.join(BASE_DB_FOLDER, "users", username)
    videos_dir = os.path.join(user_folder, "videos")
    chroma_dir = os.path.join(user_folder, "chroma_db")
    thumbnails_dir = os.path.join(user_folder, "thumbnails")  # <--- NEW

    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(chroma_dir, exist_ok=True)
    os.makedirs(thumbnails_dir, exist_ok=True)  # <--- NEW

    # Now returns 3 values, fixing the crash
    return videos_dir, chroma_dir, thumbnails_dir


def get_safe_collection_name(video_name):
    safe_hash = base64.b64encode(video_name.encode()).decode().replace("=", "").replace("/", "_").replace("+", "-")
    return f"vid_{safe_hash}"


# --- Thumbnail Generator ---
def generate_thumbnail(video_path, thumbnail_path):
    try:
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        if success:
            # Resize to save space
            frame = cv2.resize(frame, (640, 360))
            cv2.imwrite(thumbnail_path, frame)
        cap.release()
    except Exception as e:
        print(f"Thumbnail error: {e}")


# --- Status Management ---
def update_progress(username, video_name, progress_percent, stage_name):
    safe_name = get_safe_collection_name(video_name)
    status_file = os.path.join(PROCESSING_FOLDER, f"{username}_{safe_name}.json")
    status_data = {"video": video_name, "progress": progress_percent, "stage": stage_name}
    with open(status_file, "w") as f:
        json.dump(status_data, f)


def clear_progress(username, video_name):
    safe_name = get_safe_collection_name(video_name)
    status_file = os.path.join(PROCESSING_FOLDER, f"{username}_{safe_name}.json")
    if os.path.exists(status_file):
        os.remove(status_file)


def get_active_progress(username):
    active_jobs = []
    for f in os.listdir(PROCESSING_FOLDER):
        if f.startswith(f"{username}_") and f.endswith(".json"):
            try:
                with open(os.path.join(PROCESSING_FOLDER, f), "r") as file:
                    active_jobs.append(json.load(file))
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
    videos_dir, chroma_dir, thumbnails_dir = get_user_paths(username)
    client = get_db_client(chroma_dir)
    col_name = get_safe_collection_name(video_name)
    try:
        client.delete_collection(col_name)
    except:
        pass

    vid_path = os.path.join(videos_dir, video_name)
    thumb_path = os.path.join(thumbnails_dir, f"{video_name}.jpg")

    if os.path.exists(vid_path): os.remove(vid_path)
    if os.path.exists(thumb_path): os.remove(thumb_path)

    clear_progress(username, video_name)
    return True


def process_video_in_background(file_path, video_name, chroma_path, username):
    _, _, thumbnails_dir = get_user_paths(username)
    thumb_path = os.path.join(thumbnails_dir, f"{video_name}.jpg")

    # Generate thumbnail first
    generate_thumbnail(file_path, thumb_path)

    update_progress(username, video_name, 5, "Initializing AI Models...")
    try:
        model = load_whisper()
        client = get_db_client(chroma_path)
        ef = get_embedding_function()
        collection_name = get_safe_collection_name(video_name)

        try:
            client.delete_collection(collection_name)
        except:
            pass

        collection = client.create_collection(name=collection_name, embedding_function=ef)

        update_progress(username, video_name, 15, "Transcribing Audio...")
        result = model.transcribe(file_path)
        segments = result['segments']

        GROUP_SIZE = 3
        ids = []
        documents = []
        metadatas = []
        total_groups = len(segments) // GROUP_SIZE + 1

        for idx, i in enumerate(range(0, len(segments), GROUP_SIZE)):
            group = segments[i: i + GROUP_SIZE]
            if not group: continue
            combined_text = " ".join([s['text'].strip() for s in group])

            ids.append(f"{collection_name}_{i}")
            documents.append(combined_text)
            metadatas.append({
                "start_time": group[0]['start'],
                "end_time": group[-1]['end'],
                "video_name": video_name,
                "source_collection": collection_name
            })

            progress = 30 + int((idx / total_groups) * 60)
            update_progress(username, video_name, progress, "Indexing Knowledge...")

        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        update_progress(username, video_name, 100, "Done!")
        time.sleep(2)

    except Exception as e:
        print(f"Error: {e}")
        update_progress(username, video_name, 0, "Error")
    finally:
        clear_progress(username, video_name)


# --- UI Functions ---
def get_videos_list(username):
    videos_dir, _, _ = get_user_paths(username)
    if not os.path.exists(videos_dir): return []
    return [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.mov', '.avi'))]


def render_upload_page(username):
    st.title("☁️ Upload Center")
    videos_dir, chroma_dir, _ = get_user_paths(username)

    with st.container(border=True):
        uploaded_file = st.file_uploader("Select Video File", type=["mp4", "mov", "avi"])
        if uploaded_file:
            file_path = os.path.join(videos_dir, uploaded_file.name)

            if st.button("Start Processing 🚀", type="primary", use_container_width=True):
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.toast("Upload Complete! Processing started.")
                thread = threading.Thread(
                    target=process_video_in_background,
                    args=(file_path, uploaded_file.name, chroma_dir, username)
                )
                thread.start()
                time.sleep(1)
                st.rerun()


def render_library_page(username):
    st.title(f"📚 {username}'s Library")

    videos = get_videos_list(username)
    if not videos:
        st.info("Your library is empty.")
        return

    _, _, thumbnails_dir = get_user_paths(username)

    for vid in videos:
        with st.container(border=True):
            col_thumb, col_details, col_actions = st.columns([1.5, 3, 1])

            with col_thumb:
                thumb_path = os.path.join(thumbnails_dir, f"{vid}.jpg")
                if os.path.exists(thumb_path):
                    st.image(thumb_path, use_container_width=True)
                else:
                    st.markdown("🎥 **No Preview**")

            with col_details:
                st.subheader(vid)

            with col_actions:
                st.write("")
                if st.button("Open", key=f"open_{vid}", type="primary", use_container_width=True):
                    st.session_state['selected_video'] = vid
                    st.session_state['current_page'] = "Chat Workspace"
                    st.rerun()

                if st.button("Delete", key=f"del_{vid}", use_container_width=True):
                    delete_video(username, vid)
                    st.rerun()
