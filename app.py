import streamlit as st
import os
import auth
import video_processor
import query_engine

# --- Setup ---
st.set_page_config(layout="wide", page_title="Video AI System")

# Initialize Databases
auth.init_user_db()

# --- Session Management ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False


def main_app_interface():
    st.sidebar.title(f"User: {st.session_state['username']}")
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()

    # 1. Render Sidebar & Get Selected Video
    selected_video_name = video_processor.render_sidebar_ui()

    st.title("Video Knowledge Base")
    col1, col2 = st.columns([2, 1])

    # 2. Main Video Player Area
    with col1:
        video_player = st.empty()
        if selected_video_name:
            video_path = os.path.join(video_processor.VIDEOS_DIR, selected_video_name)
            video_player.video(video_path)
        else:
            st.info("Please select or upload a video.")
            video_path = None

    # 3. Search / RAG Area
    with col2:
        if selected_video_name and video_path:
            query_engine.render_search_ui(selected_video_name, video_path, video_player)


# --- Routing ---
if st.session_state['logged_in']:
    main_app_interface()
else:
    auth.render_login_ui()
