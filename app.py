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
            /* 1. Remove standard Streamlit header */
            header {visibility: hidden;}

            /* 2. Main Background - Darker Grey */
            .stApp {
                background-color: #0E1117;
            }

            /* 3. Sidebar Styling */
            section[data-testid="stSidebar"] {
                background-color: #161B22; 
                border-right: 1px solid #30363D;
            }

            /* 4. Custom Buttons (Gradient) */
            div.stButton > button:first-child {
                background: linear-gradient(to right, #FF4B4B, #FF6B6B);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
                transition: all 0.3s ease;
            }
            div.stButton > button:first-child:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 10px rgba(255, 75, 75, 0.4);
            }

            /* 5. Inputs and Text Areas */
            .stTextInput > div > div > input {
                background-color: #21262D;
                color: white;
                border-radius: 8px;
                border: 1px solid #30363D;
            }

            /* 6. Cards (Containers) */
            div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
                /* Only applies to inner containers */
            }

            /* 7. Chat Input Styling */
            .stChatInputContainer {
                padding-bottom: 20px;
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

# Initialize Secrets
if 'gemini_api_key' not in st.session_state:
    if "GEMINI_API_KEY" in st.secrets:
        st.session_state['gemini_api_key'] = st.secrets["GEMINI_API_KEY"]
    else:
        st.session_state['gemini_api_key'] = ""

auth.init_user_db()

# --- Session State ---
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'current_page' not in st.session_state: st.session_state['current_page'] = "Chat Workspace"
if 'selected_video' not in st.session_state: st.session_state['selected_video'] = None
if 'chat_history' not in st.session_state: st.session_state['chat_history'] = []


def main_app():
    username = st.session_state['username']

    # === 🎨 SIDEBAR DESIGN ===
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background: #21262D; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="margin:0; color: #E6EDF3;">⚡ PinPoint</h2>
            <p style="margin:0; font-size: 12px; color: #8B949E;">AI Video Workspace</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"👋 **Hello, {username}**")
        st.write("")  # Spacer

        # Navigation with Icons
        options = {
            "Chat Workspace": "💬",
            "Video Library": "📚",
            "Upload Center": "☁️"
        }

        # Helper to find index
        page_list = list(options.keys())
        current_idx = page_list.index(st.session_state['current_page'])

        selected_option = st.radio(
            "Go to:",
            page_list,
            index=current_idx,
            format_func=lambda x: f"{options[x]}  {x}",
            label_visibility="collapsed"
        )

        if selected_option != st.session_state['current_page']:
            st.session_state['current_page'] = selected_option
            if selected_option != "Chat Workspace":
                st.session_state['selected_video'] = None
            st.rerun()

        st.divider()

        # Settings
        if not st.session_state['gemini_api_key']:
            with st.expander("🔑 API Key Required", expanded=True):
                api_input = st.text_input("Gemini Key", type="password", help="Get from Google AI Studio")
                if api_input: st.session_state['gemini_api_key'] = api_input

        # Processing Status (Mini Widget)
        active_jobs = video_processor.get_active_progress(username)
        if active_jobs:
            st.info(f"⚡ Processing {len(active_jobs)} video(s)...")

        st.write("")
        if st.button("Logout", use_container_width=True):
            st.session_state['logged_in'] = False
            st.rerun()

    # === PAGE ROUTING ===
    if st.session_state['current_page'] == "Upload Center":
        video_processor.render_upload_page(username)

    elif st.session_state['current_page'] == "Video Library":
        video_processor.render_library_page(username)

    elif st.session_state['current_page'] == "Chat Workspace":

        # --- SCENARIO A: Global Search (RAG) ---
        if st.session_state['selected_video'] is None:

            # Hero Section
            st.markdown("""
            <div style="text-align: center; padding: 40px 0;">
                <h1 style="font-size: 3rem; background: -webkit-linear-gradient(left, #FF4B4B, #FF914D); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    Ask your videos anything.
                </h1>
                <p style="color: #8B949E; font-size: 1.2rem;">
                    Search across your entire library using advanced AI.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # 🚀 UPGRADE: Use chat_input (Sticky bottom bar)
            user_query = st.chat_input("What would you like to know?")

            # Display History (Mockup for now, or just result of last query)
            if 'previous_query' in st.session_state and st.session_state['previous_query']:
                with st.chat_message("user"):
                    st.write(st.session_state['previous_query'])

                if 'ai_answer' in st.session_state:
                    with st.chat_message("assistant", avatar="⚡"):
                        st.write(st.session_state['ai_answer'])

                        # Sources Expander
                        if st.session_state.get('global_matches'):
                            with st.expander("📚 View Sources"):
                                for idx, match in enumerate(st.session_state['global_matches']):
                                    col_src_txt, col_src_btn = st.columns([4, 1])
                                    with col_src_txt:
                                        st.caption(f"**{match['video_name']}**")
                                        st.markdown(f"*{match['text'][:150]}...*")
                                    with col_src_btn:
                                        if st.button("Play", key=f"src_{idx}"):
                                            st.session_state['selected_video'] = match['video_name']
                                            st.session_state['start_time'] = match['start_time']
                                            st.rerun()

                    # ⚠️ Disclaimer
                    st.caption("🤖 *AI responses can be inaccurate. Please verify important information.*")

            # Logic for new query
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

        # --- SCENARIO B: Watching Specific Video ---
        else:
            col_back, col_title = st.columns([1, 8])
            with col_back:
                if st.button("⬅ Back"):
                    st.session_state['selected_video'] = None
                    st.rerun()
            with col_title:
                st.subheader(f"🎬 {st.session_state['selected_video']}")

            st.divider()

            selected_vid = st.session_state['selected_video']
            videos_dir, _, _ = video_processor.get_user_paths(username)
            video_path = os.path.join(videos_dir, selected_vid)

            # Wider video player
            col_player, col_chat = st.columns([2, 1])
            with col_player:
                video_player = st.empty()
                start_ts = st.session_state.get('start_time', 0)
                video_player.video(video_path, start_time=int(start_ts))

            with col_chat:
                query_engine.render_search_ui(
                    selected_vid, video_path, video_player, username, st.session_state['gemini_api_key']
                )


if st.session_state['logged_in']:
    main_app()
else:
    auth.render_login_ui()
