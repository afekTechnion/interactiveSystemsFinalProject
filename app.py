import streamlit as st
import os
import glob
import auth
import video_processor
import query_engine

# --- Setup ---
st.set_page_config(layout="wide", page_title="PinPoint AI", page_icon="⚡")


# --- 🎨 PRO DESIGN: Custom CSS ---
def load_css():
    st.markdown("""
        <style>
            header {visibility: hidden;}
            .stApp {background-color: #0E1117;}
            section[data-testid="stSidebar"] {background-color: #161B22; border-right: 1px solid #30363D;}
            div.stButton > button:first-child {
                background: linear-gradient(to right, #FF4B4B, #FF6B6B);
                color: white; border: none; border-radius: 8px; font-weight: bold;
            }
            div.stButton > button:first-child:hover {
                transform: translateY(-2px); box-shadow: 0 4px 10px rgba(255, 75, 75, 0.4);
            }
            .stTextInput > div > div > input {
                background-color: #21262D; color: white; border-radius: 8px; border: 1px solid #30363D;
            }
        </style>
    """, unsafe_allow_html=True)


load_css()

# --- Pre-load Models ---
if 'models_loaded' not in st.session_state:
    with st.spinner("🚀 Warming up AI engines..."):
        query_engine.load_reranker()
        st.session_state['models_loaded'] = True


# --- Cleanup Logic ---
def cleanup_stuck_locks():
    if 'cleanup_done' not in st.session_state:
        lock_files = glob.glob(os.path.join(video_processor.PROCESSING_FOLDER, "*.lock"))
        for f in lock_files:
            try:
                os.remove(f)
            except:
                pass
        st.session_state['cleanup_done'] = True


cleanup_stuck_locks()

if 'gemini_api_key' not in st.session_state:
    if "GEMINI_API_KEY" in st.secrets:
        st.session_state['gemini_api_key'] = st.secrets["GEMINI_API_KEY"]
    else:
        st.session_state['gemini_api_key'] = ""

auth.init_user_db()

# --- Session State ---
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'current_page' not in st.session_state: st.session_state['current_page'] = "✨ AI Chat"
if 'selected_video' not in st.session_state: st.session_state['selected_video'] = None
if 'chat_history' not in st.session_state: st.session_state['chat_history'] = []
if 'start_time' not in st.session_state: st.session_state['start_time'] = 0


def main_app():
    username = st.session_state['username']

    # === SIDEBAR ===
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background: #21262D; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="margin:0; color: #E6EDF3;">⚡ PinPoint</h2>
            <p style="margin:0; font-size: 12px; color: #8B949E;">Pro Video Workspace</p>
        </div>
        """, unsafe_allow_html=True)

        st.caption(f"Logged in as: {username}")

        # === NAVIGATION ===
        nav_options = ["✨ AI Chat", "🎬 My Studio", "📥 Import"]

        # Fallback if state has old name
        if st.session_state['current_page'] not in nav_options:
            st.session_state['current_page'] = "✨ AI Chat"

        selected_option = st.radio(
            "Menu",
            nav_options,
            index=nav_options.index(st.session_state['current_page']),
            label_visibility="collapsed"
        )

        if selected_option != st.session_state['current_page']:
            st.session_state['current_page'] = selected_option
            if selected_option != "✨ AI Chat":
                st.session_state['selected_video'] = None
            st.rerun()

        st.divider()

        # Settings
        if not st.session_state['gemini_api_key']:
            with st.expander("🔑 API Key", expanded=True):
                api_input = st.text_input("Enter Key", type="password")
                if api_input: st.session_state['gemini_api_key'] = api_input

        # Processing Widget
        active_jobs = video_processor.get_active_progress(username)
        if active_jobs:
            st.info(f"⚡ Processing {len(active_jobs)} item(s)")

        st.write("")
        if st.button("Logout", use_container_width=True):
            st.session_state['logged_in'] = False
            st.rerun()

    # === PAGE ROUTING ===
    if st.session_state['current_page'] == "📥 Import":
        video_processor.render_upload_page(username)

    elif st.session_state['current_page'] == "🎬 My Studio":
        video_processor.render_library_page(username)

    elif st.session_state['current_page'] == "✨ AI Chat":

        # --- SCENARIO A: Global Search (No Video Selected) ---
        if st.session_state['selected_video'] is None:
            # Hero Section
            st.markdown("""
            <div style="text-align: center; padding: 40px 0;">
                <h1 style="font-size: 3rem; background: -webkit-linear-gradient(left, #FF4B4B, #FF914D); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    Ask your videos anything.
                </h1>
                <p style="color: #8B949E; font-size: 1.2rem;">
                    Your personal video intelligence hub.
                </p>
            </div>
            """, unsafe_allow_html=True)

            user_query = st.chat_input("Search across your entire library...")

            # Show Previous Query/History
            if 'previous_query' in st.session_state and st.session_state['previous_query']:
                with st.chat_message("user"):
                    st.write(st.session_state['previous_query'])

                if 'ai_answer' in st.session_state:
                    with st.chat_message("assistant", avatar="⚡"):
                        st.write(st.session_state['ai_answer'])

                        if st.session_state.get('global_matches'):
                            with st.expander("📚 View Sources"):
                                for idx, match in enumerate(st.session_state['global_matches']):
                                    c1, c2 = st.columns([4, 1])
                                    with c1:
                                        st.caption(f"**{match['video_name']}**: *{match['text'][:100]}...*")
                                    with c2:
                                        if st.button("Play", key=f"src_{idx}"):
                                            st.session_state['selected_video'] = match['video_name']
                                            st.session_state['start_time'] = match['start_time']
                                            st.rerun()

            if user_query:
                st.session_state['previous_query'] = user_query
                with st.spinner("🧠 Thinking..."):
                    matches = query_engine.search_all_collections(user_query, username)
                    st.session_state['global_matches'] = matches
                    if matches:
                        ai_response = query_engine.ask_gemini(user_query, matches, st.session_state['gemini_api_key'])
                        st.session_state['ai_answer'] = ai_response
                    else:
                        st.session_state['ai_answer'] = "I couldn't find any relevant information in your library."
                st.rerun()

        # --- SCENARIO B: Watching Specific Video (FIXED) ---
        else:
            col_back, col_title = st.columns([1, 8])
            with col_back:
                if st.button("⬅ Back"):
                    st.session_state['selected_video'] = None
                    st.session_state['start_time'] = 0
                    st.rerun()
            with col_title:
                st.subheader(f"🎬 {st.session_state['selected_video']}")

            st.divider()

            selected_vid = st.session_state['selected_video']
            videos_dir, _, _ = video_processor.get_user_paths(username)
            video_path = os.path.join(videos_dir, selected_vid)

            col_player, col_chat = st.columns([2, 1])
            with col_player:
                video_player = st.empty()
                start_ts = st.session_state.get('start_time', 0)

                # --- TWEAKED FIX: REMOVED 'key' (Prevents crash) ---
                # Added .empty() above to try and clear the previous video
                video_player.empty()
                video_player.video(
                    video_path,
                    start_time=int(start_ts)
                )

            with col_chat:
                query_engine.render_search_ui(
                    selected_vid, video_path, video_player, username, st.session_state['gemini_api_key']
                )


if st.session_state['logged_in']:
    main_app()
else:
    auth.render_login_ui()