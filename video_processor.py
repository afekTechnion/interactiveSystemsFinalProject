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
    thumbnails_dir = os.path.join(user_folder, "thumbnails")

    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(chroma_dir, exist_ok=True)
    os.makedirs(thumbnails_dir, exist_ok=True)

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


def create_completion_notification(username, video_name):
    """Creates a temporary file to signal the frontend that a job is done."""
    safe_name = get_safe_collection_name(video_name)
    # Create a file named "done_username_videoname.flag"
    note_file = os.path.join(PROCESSING_FOLDER, f"done_{username}_{safe_name}.flag")
    with open(note_file, 'w') as f:
        f.write(video_name)


def get_and_clear_notifications(username):
    """Checks for completion flags, returns the video names, and deletes the flags."""
    completed_videos = []
    # Search for files starting with "done_username_"
    for f in os.listdir(PROCESSING_FOLDER):
        if f.startswith(f"done_{username}_") and f.endswith(".flag"):
            path = os.path.join(PROCESSING_FOLDER, f)
            try:
                with open(path, 'r') as file:
                    completed_videos.append(file.read())
                os.remove(path) # Delete immediately so we only notify once
            except:
                pass
    return completed_videos


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
        create_completion_notification(username, video_name)
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


# === 🎨 NEW DESIGN: Upload Page with Progress ===
def render_upload_page(username):
    st.title("📥 Import Content")

    # --- PROGESS SECTION START ---
    active_jobs = get_active_progress(username)
    if active_jobs:
        st.info("🔄 Processing in background...")
        for job in active_jobs:
            st.write(f"**{job['video']}**: {job['stage']}")
            st.progress(job['progress'])

        # This causes the page to refresh, animating the bar
        time.sleep(1)
        st.rerun()
    # --- PROGRESS SECTION END ---

    videos_dir, chroma_dir, _ = get_user_paths(username)

    # 1. Fake Storage Status Bar (Aesthetic)
    col_stat1, col_stat2 = st.columns([3, 1])
    with col_stat1:
        st.progress(45, text="Cloud Storage Usage (Demo)")
    with col_stat2:
        st.caption("🚀 4.5GB / 10GB Used")

    st.divider()

    # 2. Upload Area
    with st.container(border=True):
        st.markdown("### 📤 Drag & Drop Video")
        uploaded_file = st.file_uploader("", type=["mp4", "mov", "avi"], label_visibility="collapsed")

        if uploaded_file:
            st.info(f"Ready to process: **{uploaded_file.name}**")

            if st.button("Start Processing ⚡", type="primary", use_container_width=True):
                file_path = os.path.join(videos_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.toast("Upload Complete! AI Processing started.")
                thread = threading.Thread(
                    target=process_video_in_background,
                    args=(file_path, uploaded_file.name, chroma_dir, username)
                )
                thread.start()
                time.sleep(1)
                st.rerun()


# === 🎨 NEW DESIGN: Library Page (Grid Layout) ===
def render_library_page(username):
    st.title("🎬 My Studio")

    videos = get_videos_list(username)
    _, _, thumbnails_dir = get_user_paths(username)

    if not videos:
        st.info("Your library is empty. Go to 'Import' to add videos.")
        return

    # --- GRID LAYOUT LOGIC ---
    cols_per_row = 3
    rows = [videos[i:i + cols_per_row] for i in range(0, len(videos), cols_per_row)]

    for row_videos in rows:
        cols = st.columns(cols_per_row)
        for idx, vid in enumerate(row_videos):
            with cols[idx]:
                # Card Container
                with st.container(border=True):
                    # 1. THUMBNAIL / PLACEHOLDER (Fixed Height)
                    thumb_path = os.path.join(thumbnails_dir, f"{vid}.jpg")

                    # Common CSS: Forces 180px height and "cover" crop for all images
                    # This ensures specific alignment regardless of video shape
                    style_settings = "width: 100%; height: 180px; object-fit: cover; border-radius: 4px; margin-bottom: 10px;"

                    if os.path.exists(thumb_path):
                        # Read and encode image to allow custom HTML styling
                        try:
                            with open(thumb_path, "rb") as img_file:
                                b64_data = base64.b64encode(img_file.read()).decode()
                            st.markdown(
                                f'<img src="data:image/jpeg;base64,{b64_data}" style="{style_settings}">',
                                unsafe_allow_html=True
                            )
                        except Exception:
                            # Fallback if read fails
                            st.markdown(
                                f'<div style="{style_settings} background-color: #262730; display:flex; align-items:center; justify-content:center; color:white;">Error</div>',
                                unsafe_allow_html=True
                            )
                    else:
                        # Placeholder with EXACT same dimensions
                        st.markdown(
                            f'<div style="{style_settings} background-color: #262730; display:flex; align-items:center; justify-content:center; color:#8B949E;">No Preview</div>',
                            unsafe_allow_html=True
                        )

                    # 2. TITLE
                    display_name = vid if len(vid) < 20 else vid[:17] + "..."
                    st.markdown(f"**{display_name}**")

                    # 3. ACTIONS
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        if st.button("Open", key=f"open_{vid}", type="secondary", use_container_width=True):
                            st.session_state['selected_video'] = vid
                            st.session_state['current_page'] = "✨ AI Chat"
                            st.rerun()

                    with c2:
                        if st.button("🗑️", key=f"del_{vid}", use_container_width=True):
                            delete_video(username, vid)
                            st.rerun()