import os
import threading
import time
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
def set_processing_status(username, video_name, status=True):
    """יוצר או מוחק קובץ מעקב"""
    safe_name = get_safe_collection_name(video_name)
    # שם הקובץ יכיל גם את שם המשתמש כדי למנוע בלבול
    lock_file = os.path.join(PROCESSING_FOLDER, f"{username}_{safe_name}.lock")

    if status:
        with open(lock_file, "w") as f:
            f.write("processing")
    else:
        if os.path.exists(lock_file):
            os.remove(lock_file)


def get_processing_videos(username):
    """מחזיר רשימה של סרטונים שנמצאים כרגע בעיבוד"""
    processing_files = [f for f in os.listdir(PROCESSING_FOLDER) if
                        f.startswith(f"{username}_") and f.endswith(".lock")]
    # נחלץ את שם הסרטון המקורי? כרגע נחזיר פשוט שיש משהו בעבודה
    return len(processing_files)


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

    # אם נמחק בזמן עיבוד - ננקה גם את הסטטוס
    set_processing_status(username, video_name, False)
    return True


def process_video_in_background(file_path, video_name, chroma_path, username):
    # 1. סימון שהעבודה התחילה
    set_processing_status(username, video_name, True)

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

        result = model.transcribe(file_path)
        segments = result['segments']

        GROUP_SIZE = 3
        ids = []
        documents = []
        metadatas = []

        for i in range(0, len(segments), GROUP_SIZE):
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

        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        print(f"Finished processing {video_name}")

    except Exception as e:
        print(f"Error processing video: {e}")

    finally:
        # 2. סימון שהעבודה נגמרה (גם אם הייתה שגיאה)
        set_processing_status(username, video_name, False)


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
                # בדיקה אם הקובץ כבר בעבודה
                if get_processing_videos(username) > 0:
                    st.warning("Wait for the current video to finish!")
                elif not os.path.exists(file_path):
                    # Upload in small 5MB chunks to prevent RAM overuse
                    with open(file_path, "wb") as f:
                        while True:
                            # Read the file in 5MB chunks (5 * 1024 * 1024 bytes)
                            chunk = uploaded_file.read(5 * 1024 * 1024)
                            if not chunk:
                                break
                            f.write(chunk)

                    st.success(f"File saved to {username}'s library")

                    # הוספנו את ה-username לפרמטרים
                    thread = threading.Thread(
                        target=process_video_in_background,
                        args=(file_path, uploaded_file.name, chroma_dir, username)
                    )
                    thread.start()
                    st.rerun()  # רענון כדי שהסטטוס יופיע מיד
                else:
                    st.warning("File already exists.")


def render_library_page(username):
    st.header(f"📚 {username}'s Library")

    # בדיקה אם יש עיבוד ברקע - אם כן נוסיף הודעה
    processing_count = get_processing_videos(username)
    if processing_count > 0:
        st.info(f"🔄 Currently processing {processing_count} video(s)... Refresh later to see them.")

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
                if st.button("🗑️", key=f"del_{vid}", help="Delete video permanently"):
                    if delete_video(username, vid):
                        st.success(f"Deleted {vid}")
                        st.rerun()