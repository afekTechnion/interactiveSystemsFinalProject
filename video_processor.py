import os
import threading
import streamlit as st
import whisper
import chromadb
from chromadb.utils import embedding_functions
import torch
import base64  # חשוב: הוספנו את זה לטיפול בשמות

# --- Configuration ---
BASE_DB_FOLDER = "Database"
device = "cuda" if torch.cuda.is_available() else "cpu"


# --- Helper: User Paths ---
def get_user_paths(username):
    """מייצר נתיבים ייחודיים לכל משתמש"""
    user_folder = os.path.join(BASE_DB_FOLDER, "users", username)
    videos_dir = os.path.join(user_folder, "videos")
    chroma_dir = os.path.join(user_folder, "chroma_db")

    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(chroma_dir, exist_ok=True)

    return videos_dir, chroma_dir


def get_safe_collection_name(video_name):
    """פונקציית עזר ליצירת שם ייחודי ל-Collection"""
    # המרת שם הקובץ לקוד מוצפן כדי למנוע בעיות עם תווים מיוחדים
    safe_hash = base64.b64encode(video_name.encode()).decode().replace("=", "").replace("/", "_").replace("+", "-")
    return f"vid_{safe_hash}"


# --- Backend Logic ---
@st.cache_resource
def load_whisper():
    return whisper.load_model("small", device=device)


def get_db_client(chroma_path):
    return chromadb.PersistentClient(path=chroma_path)


def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def delete_video(username, video_name):
    """מוחק את הוידאו מהתיקייה ומהדאטה בייס"""
    videos_dir, chroma_dir = get_user_paths(username)
    client = get_db_client(chroma_dir)

    # 1. מחיקה מהדאטה בייס
    col_name = get_safe_collection_name(video_name)
    try:
        client.delete_collection(col_name)
        print(f"Deleted collection: {col_name}")
    except ValueError:
        print(f"Collection {col_name} not found or already deleted")
    except Exception as e:
        print(f"Error deleting collection: {e}")

    # 2. מחיקה פיזית של הקובץ
    file_path = os.path.join(videos_dir, video_name)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return True
        except OSError as e:
            print(f"Error deleting file: {e}")
            return False
    return True


def process_video_in_background(file_path, video_name, chroma_path):
    try:
        model = load_whisper()
        client = get_db_client(chroma_path)
        ef = get_embedding_function()

        # שימוש בפונקציית העזר לשם בטוח
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
                if not os.path.exists(file_path):
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    st.success(f"File saved to {username}'s library")

                    thread = threading.Thread(
                        target=process_video_in_background,
                        args=(file_path, uploaded_file.name, chroma_dir)
                    )
                    thread.start()
                    st.info("Processing started in background!")
                else:
                    st.warning("File already exists.")


def render_library_page(username):
    st.header(f"📚 {username}'s Library")

    videos = get_videos_list(username)

    if not videos:
        st.info("Your library is empty.")
        return

    search = st.text_input("Filter library...", "")
    filtered_videos = [v for v in videos if search.lower() in v.lower()]

    for vid in filtered_videos:
        with st.container(border=True):
            # שינינו את החלוקה לעמודות כדי לפנות מקום לכפתור המחיקה
            col_text, col_open, col_del = st.columns([5, 1.5, 0.5])

            with col_text:
                st.subheader(f"🎬 {vid}")

            with col_open:
                st.write("")  # Spacer
                if st.button("Open Workspace", key=f"btn_{vid}", use_container_width=True):
                    st.session_state['selected_video'] = vid
                    st.session_state['current_page'] = "Chat Workspace"
                    st.rerun()

            with col_del:
                st.write("")  # Spacer
                # כפתור מחיקה אדום וקטן
                if st.button("🗑️", key=f"del_{vid}", help="Delete video permanently"):
                    if delete_video(username, vid):
                        st.success(f"Deleted {vid}")
                        st.rerun()
                    else:
                        st.error("Failed to delete")