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
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "Chat Workspace"
if 'selected_video' not in st.session_state:
    st.session_state['selected_video'] = None
if 'gemini_api_key' not in st.session_state:
    st.session_state['gemini_api_key'] = ""


def main_app():
    # === SIDEBAR ===
    with st.sidebar:
        st.title(f"👤 {st.session_state['username']}")

        # Navigation
        menu_selection = st.radio(
            "Navigation",
            ["Chat Workspace", "Video Library", "Upload Center"],
            index=["Chat Workspace", "Video Library", "Upload Center"].index(st.session_state['current_page'])
        )
        if menu_selection != st.session_state['current_page']:
            st.session_state['current_page'] = menu_selection
            if menu_selection != "Chat Workspace":
                st.session_state['selected_video'] = None
            st.rerun()

        st.markdown("---")

        # API Key Input
        with st.expander("⚙️ AI Settings", expanded=False):
            st.caption("Required for RAG Answer Generation")
            api_key_input = st.text_input("Gemini API Key", type="password", value=st.session_state['gemini_api_key'])
            if api_key_input:
                st.session_state['gemini_api_key'] = api_key_input

        st.markdown("---")
        if st.button("Logout", icon="🚪"):
            st.session_state['logged_in'] = False
            st.rerun()

    # === PAGE ROUTING ===
    if st.session_state['current_page'] == "Upload Center":
        video_processor.render_upload_page()

    elif st.session_state['current_page'] == "Video Library":
        video_processor.render_library_page()

    # === RAG CHAT WORKSPACE ===
    elif st.session_state['current_page'] == "Chat Workspace":

        # Scenario A: Global Search (The RAG Part)
        if st.session_state['selected_video'] is None:
            st.header("🧠 Knowledge Query")

            if 'last_query' not in st.session_state:
                st.session_state['last_query'] = ""

            user_query = st.text_input("Ask your video library:", value=st.session_state['last_query'])

            if user_query and user_query != st.session_state.get('previous_query', ''):
                st.session_state['previous_query'] = user_query

                with st.spinner("Analyzing videos & Generating answer..."):
                    # 1. Retrieve & Rerank (Get the Sources)
                    matches = query_engine.search_all_collections(user_query)
                    st.session_state['global_matches'] = matches

                    # 2. Generation (Ask Gemini)
                    if matches:
                        ai_response = query_engine.ask_gemini(
                            user_query,
                            matches,
                            st.session_state['gemini_api_key']
                        )
                        st.session_state['ai_answer'] = ai_response
                    else:
                        st.session_state['ai_answer'] = "No relevant videos found to answer this question."

            # Display Results
            if user_query:
                # 1. The AI Answer
                st.subheader("🤖 AI Answer")
                if 'ai_answer' in st.session_state:
                    # הוספתי כאן color: black; כדי שהטקסט יהיה קריא
                    st.markdown(f"""
                                    <div style="background-color: #f0f2f6; color: black; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b;">
                                        {st.session_state['ai_answer']}
                                    </div>
                                    """, unsafe_allow_html=True)

                st.divider()

                # 2. The Source Videos (Citations)
                st.subheader("📚 Source Videos")
                matches = st.session_state.get('global_matches', [])

                if not matches:
                    st.warning("No matches found.")

                for idx, match in enumerate(matches):
                    with st.container(border=True):
                        col_info, col_action = st.columns([4, 1])
                        with col_info:
                            st.markdown(f"**🎬 Source:** `{match['video_name']}`")
                            st.caption(f"Relevant Excerpt: \"{match['reason'][:200]}...\"")
                        with col_action:
                            st.write("")
                            if st.button("Watch", key=f"select_{idx}", type="primary", use_container_width=True):
                                st.session_state['selected_video'] = match['video_name']
                                st.session_state['start_time'] = match['start_time']
                                st.rerun()

        # Scenario B: Watching a specific video
        else:
            col_back, col_title = st.columns([1, 5])
            with col_back:
                if st.button("⬅ Back"):
                    st.session_state['selected_video'] = None
                    st.rerun()
            with col_title:
                st.subheader(f"Watching: {st.session_state['selected_video']}")

            st.divider()

            selected_vid = st.session_state['selected_video']
            video_path = os.path.join(video_processor.VIDEOS_DIR, selected_vid)

            col_player, col_chat = st.columns([1.5, 1])
            with col_player:
                video_player = st.empty()
                start_ts = st.session_state.get('start_time', 0)
                video_player.video(video_path, start_time=int(start_ts))
            with col_chat:
                query_engine.render_search_ui(selected_vid, video_path, video_player)


if st.session_state['logged_in']:
    main_app()
else:
    auth.render_login_ui()
