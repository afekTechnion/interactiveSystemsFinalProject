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
        st.title(f"Hello {st.session_state['username']}👤")
        st.markdown("---")

        menu_selection = st.radio(
            "Navigation",
            ["Chat Workspace", "Video Library", "Upload Center"],
            index=["Chat Workspace", "Video Library", "Upload Center"].index(st.session_state['current_page'])
        )

        if menu_selection != st.session_state['current_page']:
            st.session_state['current_page'] = menu_selection
            st.rerun()

        st.markdown("---")
        if st.button("Logout", icon="🚪"):
            st.session_state['logged_in'] = False
            st.rerun()

    # === PAGE ROUTING ===

    if st.session_state['current_page'] == "Upload Center":
        video_processor.render_upload_page()

    elif st.session_state['current_page'] == "Video Library":
        video_processor.render_library_page()

    # === NEW CHAT WORKSPACE LOGIC ===
    elif st.session_state['current_page'] == "Chat Workspace":
        st.header("🧠 Knowledge Query")

        # 1. THE GLOBAL SEARCH BAR
        # We store the search query in session state so it doesn't disappear when we click buttons
        if 'last_query' not in st.session_state:
            st.session_state['last_query'] = ""

        user_query = st.text_input("Ask a question across all videos:", value=st.session_state['last_query'])

        # If the user typed something new, update the results
        if user_query and user_query != st.session_state.get('previous_query', ''):
            st.session_state['global_matches'] = query_engine.search_all_collections(user_query)
            st.session_state['previous_query'] = user_query
            st.session_state['selected_video'] = None  # Reset selection on new search

        # 2. DISPLAY TOP 3 RESULTS (If no video is selected yet)
        if user_query and not st.session_state['selected_video']:
            st.subheader("Top 3 Relevant Videos Found:")

            matches = st.session_state.get('global_matches', [])

            if not matches:
                st.warning("No relevant answers found in your library.")

            for idx, match in enumerate(matches):
                # Create a card for each result
                with st.container(border=True):
                    col_info, col_action = st.columns([4, 1])

                    with col_info:
                        st.markdown(f"**🎬 Video Source:** `{match['video_name']}`")
                        # Show the specific sentence that matched ("The Reason")
                        st.info(f"💡 **Context Found:** \"{match['reason']}\"")

                    with col_action:
                        st.write("")  # Spacer
                        # The button that selects this video
                        if st.button("Select & Watch", key=f"select_{idx}", type="primary"):
                            st.session_state['selected_video'] = match['video_name']
                            # Optional: We can also auto-jump to the timestamp!
                            st.session_state['start_time'] = match['start_time']
                            st.rerun()

        # 3. THE WORKSPACE (Only shows AFTER a video is selected)
        if st.session_state['selected_video']:
            st.divider()

            # Back button to clear selection
            if st.button("⬅ Back to Search Results"):
                st.session_state['selected_video'] = None
                st.rerun()

            selected_vid = st.session_state['selected_video']
            st.subheader(f"Working on: {selected_vid}")

            col1, col2 = st.columns([1.5, 1])

            with col1:
                video_path = os.path.join(video_processor.VIDEOS_DIR, selected_vid)
                video_player = st.empty()

                # Auto-jump if we came from a search result
                start_ts = st.session_state.get('start_time', 0)
                video_player.video(video_path, start_time=int(start_ts))

            with col2:
                # Allow searching AGAIN within this specific video
                query_engine.render_search_ui(selected_vid, video_path, video_player)


# --- Entry Point ---
if st.session_state['logged_in']:
    main_app()
else:
    auth.render_login_ui()
