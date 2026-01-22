import streamlit as st
import os
import glob
import auth
import video_processor
import query_engine

# --- Setup ---
st.set_page_config(
    layout="wide",
    page_title="PinPoint AI",
    page_icon="⚡",
    initial_sidebar_state="expanded"  # <--- This forces it open
)


# --- 🎨 PRO DESIGN: Custom CSS ---
def load_css():
    st.markdown("""
        <style>
            /* 1. Set Background Colors */
            .stApp {background-color: #0E1117;}
            section[data-testid="stSidebar"] {background-color: #161B22; border-right: 1px solid #30363D;}

            /* 2. Style Buttons (Red Gradient) */
            div.stButton > button:first-child {
                background: linear-gradient(to right, #FF4B4B, #FF6B6B);
                color: white; 
                border: none; 
                border-radius: 8px; 
                font-weight: bold;
            }

            /* 3. Style Inputs (Dark Grey) */
            .stTextInput > div > div > input {
                background-color: #21262D; 
                color: white; 
                border-radius: 8px; 
                border: 1px solid #30363D;
            }

            /* 4. DO NOT HIDE THE HEADER YET */
            /* We leave the header visible to ensure the button appears. */
        </style>
    """, unsafe_allow_html=True)


load_css()

# --- Pre-load Models ---
if 'models_loaded' not in st.session_state:
    with st.spinner("🚀 Warming up AI engines..."):
        query_engine.load_reranker()
        st.session_state['models_loaded'] = True


# --- Cleanup ---
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

# --- NEW: UI LOCK STATE ---
if 'processing_global' not in st.session_state: st.session_state['processing_global'] = False


def lock_global_chat():
    """Callback to lock input immediately when Enter is pressed"""
    st.session_state['processing_global'] = True


def main_app():
    username = st.session_state['username']

    # === NEW: CHECK FOR COMPLETED VIDEOS ===
    # This checks if any background jobs finished since the last update
    completed_jobs = video_processor.get_and_clear_notifications(username)
    if completed_jobs:
        for video_name in completed_jobs:
            # Show a nice popup notification
            st.toast(f"✅ Processing Complete: **{video_name}**", icon="🎉")
    # =======================================

    # === FORCE SIDEBAR RENDER ===
    # We open the sidebar context immediately to ensure Streamlit draws the frame.
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background: #21262D; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="margin:0; color: #E6EDF3;">⚡ PinPoint</h2>
            <p style="margin:0; font-size: 12px; color: #8B949E;">Pro Video Workspace</p>
        </div>
        """, unsafe_allow_html=True)

        st.caption(f"👤 {username}")

        # Navigation
        nav_options = ["✨ AI Chat", "🎬 My Studio", "📥 Import"]

        # Ensure valid page selection
        if st.session_state.get('current_page') not in nav_options:
            st.session_state['current_page'] = "✨ AI Chat"

        selected_option = st.radio(
            "Menu", nav_options,
            index=nav_options.index(st.session_state['current_page']),
            label_visibility="collapsed"
        )

        if selected_option != st.session_state['current_page']:
            st.session_state['current_page'] = selected_option
            # Clear selected video if leaving chat
            if selected_option != "✨ AI Chat":
                st.session_state['selected_video'] = None
            st.rerun()

        st.divider()

        if not st.session_state['gemini_api_key']:
            with st.expander("🔑 API Key", expanded=True):
                api_input = st.text_input("Enter Key", type="password")
                if api_input: st.session_state['gemini_api_key'] = api_input

            # --- SAFE PROGRESS CHECK ---
        try:
            active_jobs = video_processor.get_active_progress(username)
            if active_jobs:
                st.info(f"⚡ Processing {len(active_jobs)} item(s)")
        except Exception as e:
            # If this fails, just pass so the sidebar doesn't crash
            pass

        st.write("")
        if st.button("Logout", use_container_width=True):
            st.session_state['logged_in'] = False
            st.rerun()

    # === MAIN CONTENT ROUTING ===
    # This runs AFTER the sidebar is drawn
    if st.session_state['current_page'] == "📥 Import":
        video_processor.render_upload_page(username)

    elif st.session_state['current_page'] == "🎬 My Studio":
        video_processor.render_library_page(username)

    elif st.session_state['current_page'] == "✨ AI Chat":
        # ... (Your existing Chat Logic here) ...
        # Copy the exact Chat Logic from your previous file
        # I will summarize the structure below to keep it short:

        if st.session_state['selected_video'] is None:
            if not st.session_state['chat_history']:
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

            # 1. RENDER HISTORY
            for i, msg in enumerate(st.session_state['chat_history']):
                with st.chat_message(msg['role'], avatar="⚡" if msg['role'] == "assistant" else None):
                    st.write(msg['content'])
                    if "sources" in msg and msg['sources']:
                        with st.expander("📚 Sources"):
                            for idx, match in enumerate(msg['sources']):
                                c1, c2 = st.columns([4, 1])
                                with c1:
                                    # Safe text truncation to avoid errors
                                    txt = match.get('text', '')[:100]
                                    st.caption(f"**{match['video_name']}**: *{txt}...*")
                                with c2:
                                    if st.button("Play", key=f"hist_{i}_{idx}"):
                                        st.session_state['selected_video'] = match['video_name']
                                        st.session_state['start_time'] = match['start_time']
                                        st.rerun()

            # 2. INPUT AREA (LOCKED WHEN PROCESSING)
            user_query = st.chat_input(
                "Search across your entire library...",
                on_submit=lock_global_chat,
                disabled=st.session_state['processing_global']
            )

            if user_query:
                # Append user message immediately
                st.session_state['chat_history'].append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.write(user_query)

                # --- THE FIX: TRY/FINALLY BLOCK ---
                try:
                    with st.chat_message("assistant", avatar="⚡"):
                        with st.spinner("🧠 Thinking..."):
                            # 1. Search
                            matches = query_engine.search_all_collections(user_query, username)

                            # 2. Generate Answer
                            if matches:
                                ai_response = query_engine.ask_gemini(user_query, matches,
                                                                      st.session_state['gemini_api_key'])
                                st.session_state['chat_history'].append({
                                    "role": "assistant",
                                    "content": ai_response,
                                    "sources": matches
                                })
                                st.write(ai_response)  # Show answer immediately
                            else:
                                msg = "I couldn't find any relevant information in your library."
                                st.session_state['chat_history'].append({"role": "assistant", "content": msg})
                                st.write(msg)

                except Exception as e:
                    st.error(f"An error occurred: {e}")

                finally:
                    # THIS ALWAYS RUNS: Unlocks the chat even if there was a crash
                    st.session_state['processing_global'] = False
                    st.rerun()

        else:
            # Watching Video Logic
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
