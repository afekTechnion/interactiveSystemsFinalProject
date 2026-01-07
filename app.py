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


# --- Main App Logic ---
def main_app():
    # === SIDEBAR NAVIGATION ===
    with st.sidebar:
        st.title(f"👤 {st.session_state['username']}")
        st.markdown("---")

        # Navigation Menu
        menu_selection = st.radio(
            "Navigation",
            ["Chat Workspace", "Video Library", "Upload Center"],
            index=["Chat Workspace", "Video Library", "Upload Center"].index(st.session_state['current_page'])
        )

        # Update page selection
        if menu_selection != st.session_state['current_page']:
            st.session_state['current_page'] = menu_selection
            # If leaving workspace, clear selection to avoid confusion later
            if menu_selection != "Chat Workspace":
                st.session_state['selected_video'] = None
            st.rerun()

        st.markdown("---")
        if st.button("Logout", icon="🚪"):
            st.session_state['logged_in'] = False
            st.rerun()

    # === PAGE ROUTING ===

    # 1. Upload Page
    if st.session_state['current_page'] == "Upload Center":
        video_processor.render_upload_page()

    # 2. Library Page
    elif st.session_state['current_page'] == "Video Library":
        video_processor.render_library_page()

    # 3. Chat Workspace (The Fixed Logic)
    elif st.session_state['current_page'] == "Chat Workspace":

        # --- SCENARIO A: No Video Selected (Show Global Search) ---
        if st.session_state['selected_video'] is None:
            st.header("🧠 Knowledge Query")
            st.write("Ask a question to find the most relevant video in your library.")

            # Global Search Bar
            if 'last_query' not in st.session_state:
                st.session_state['last_query'] = ""

            user_query = st.text_input("Global Search:", value=st.session_state['last_query'],
                                       placeholder="e.g., What did we discuss about Q3 earnings?")

            # Process Search
            if user_query and user_query != st.session_state.get('previous_query', ''):
                with st.spinner("Scanning library..."):
                    st.session_state['global_matches'] = query_engine.search_all_collections(user_query)
                    st.session_state['previous_query'] = user_query

            # Display Results
            if user_query:
                st.subheader("Top Matches")
                matches = st.session_state.get('global_matches', [])

                if not matches:
                    st.warning("No relevant videos found.")

                for idx, match in enumerate(matches):
                    with st.container(border=True):
                        col_info, col_action = st.columns([4, 1])
                        with col_info:
                            st.markdown(f"**🎬 Source:** `{match['video_name']}`")
                            st.info(f"💡 \"{match['reason']}\"")
                        with col_action:
                            st.write("")
                            if st.button("Select", key=f"select_{idx}", type="primary", use_container_width=True):
                                st.session_state['selected_video'] = match['video_name']
                                st.session_state['start_time'] = match['start_time']
                                st.rerun()

        # --- SCENARIO B: Video Selected (Show Player + Local Chat ONLY) ---
        else:
            # Header with Back Button
            col_back, col_title = st.columns([1, 5])
            with col_back:
                if st.button("⬅ Back to Search"):
                    st.session_state['selected_video'] = None
                    st.rerun()
            with col_title:
                st.subheader(f"Watching: {st.session_state['selected_video']}")

            st.divider()

            # The Workspace Layout
            selected_vid = st.session_state['selected_video']
            video_path = os.path.join(video_processor.VIDEOS_DIR, selected_vid)

            col_player, col_chat = st.columns([1.5, 1])

            with col_player:
                video_player = st.empty()
                # Jump to specific time if provided
                start_ts = st.session_state.get('start_time', 0)
                video_player.video(video_path, start_time=int(start_ts))

            with col_chat:
                # This renders the "Local Search" input box
                query_engine.render_search_ui(selected_vid, video_path, video_player)


# --- Entry Point ---
if st.session_state['logged_in']:
    main_app()
else:
    auth.render_login_ui()
