import streamlit as st
import video_processor
import torch
import re
from sentence_transformers import CrossEncoder
import google.generativeai as genai

# --- CONFIGURATION ---
GEMINI_MODEL_NAME = "gemini-2.5-flash"
INITIAL_TOP_K = 10
FINAL_TOP_K = 3


@st.cache_resource(show_spinner=False)
def load_reranker():
    if torch.cuda.is_available():
        device = "cuda"
        print("\n✅ GPU DETECTED: RUNNING IN FAST MODE\n")
    else:
        device = "cpu"
        print("\n⚠️ GPU NOT FOUND: RUNNING IN SLOW CPU MODE\n")
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)


def expand_context(collection, center_id, window=1):
    try:
        base_name, index_str = center_id.rsplit('_', 1)
        current_idx = int(index_str)
        ids_to_fetch = []
        for i in range(current_idx - window, current_idx + window + 1):
            if i >= 0: ids_to_fetch.append(f"{base_name}_{i}")
        data = collection.get(ids=ids_to_fetch)
        sorted_docs = sorted(zip(data['ids'], data['documents']), key=lambda x: int(x[0].rsplit('_', 1)[1]))
        full_text = " ".join([doc for _, doc in sorted_docs])
        return full_text
    except Exception:
        return ""


def search_single_video(collection_name, query_text, username, n_results=5):
    _, chroma_dir, _ = video_processor.get_user_paths(username)
    client = video_processor.get_db_client(chroma_dir)
    try:
        collection = client.get_collection(collection_name)
        return collection.query(query_texts=[query_text], n_results=n_results)
    except ValueError:
        return None


def search_all_collections(query_text, username):
    videos_dir, chroma_dir, _ = video_processor.get_user_paths(username)
    client = video_processor.get_db_client(chroma_path=chroma_dir)
    videos = video_processor.get_videos_list(username)
    reranker = load_reranker()
    initial_candidates = []

    for video_name in videos:
        col_name = video_processor.get_safe_collection_name(video_name)
        try:
            collection = client.get_collection(col_name)
            results = collection.query(query_texts=[query_text], n_results=2)
            if results['documents']:
                for i in range(len(results['documents'][0])):
                    doc_id = results['ids'][0][i]
                    expanded_text = expand_context(collection, doc_id, window=1)
                    meta = results['metadatas'][0][i]
                    initial_candidates.append({
                        "video_name": video_name,
                        "text": expanded_text,
                        "start_time": meta['start_time']
                    })
        except Exception:
            continue

    if not initial_candidates: return []

    rerank_pairs = [[query_text, candidate['text']] for candidate in initial_candidates]
    scores = reranker.predict(rerank_pairs)

    for i, candidate in enumerate(initial_candidates):
        candidate['score'] = scores[i]
        candidate['reason'] = candidate['text']

    initial_candidates.sort(key=lambda x: x['score'], reverse=True)
    return initial_candidates[:FINAL_TOP_K]


def format_local_fallback(query, context_results, error_msg):
    response = f"**⚠️ Cloud AI Unavailable ({error_msg})**\n\n"
    response += "Using **Local Fallback Mode**. Matches found:\n\n"
    for i, item in enumerate(context_results):
        response += f"**{i + 1}. {item.get('video_name', 'Video')}**\n> *\"{item['text']}\"*\n\n"
    return response


def ask_gemini(query, context_results, api_key):
    if not api_key: return format_local_fallback(query, context_results, "No API Key")
    try:
        genai.configure(api_key=api_key)
        context_text = ""
        for item in context_results: context_text += f"- {item['text']}\n"
        prompt = f"""
                You are a helpful AI study assistant.

                INSTRUCTIONS:
                1. Answer the User Question primarily using the provided Context from the video.
                2. You may elaborate or explain concepts further using your own knowledge to be more helpful.
                3. CRITICAL: If the provided Context does NOT answer the question, answer based on your general knowledge, BUT you must start your answer with: "I couldn't find this in the video, but generally speaking..."

                Context:
                {context_text}

                User Question: {query}

                Answer:
                """

        try:
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            return model.generate_content(prompt).text
        except Exception:
            try:
                model = genai.GenerativeModel('gemini-pro')
                return model.generate_content(prompt).text
            except Exception as e:
                return format_local_fallback(query, context_results, "Connection Failed")
    except Exception as e:
        return format_local_fallback(query, context_results, str(e))


# --- LOCK CALLBACK ---
def lock_video_chat():
    st.session_state['processing_video'] = True


def highlight_text(text, query):
    if not query: return text[:100] + "..."  # Return short snippet if no query

    words = query.lower().split()

    # 1. SMART TRUNCATION: Find the first keyword and cut around it
    first_match_index = -1
    for w in words:
        if len(w) > 2:
            idx = text.lower().find(w)
            if idx != -1:
                first_match_index = idx
                break

    if first_match_index != -1:
        # Start 30 chars before match, End 60 chars after match
        start = max(0, first_match_index - 30)
        end = min(len(text), first_match_index + 60)
        snippet = "..." + text[start:end] + "..."
    else:
        # Fallback if no match found (rare)
        snippet = text[:80] + "..."

    # 2. FADED HIGHLIGHTING
    highlighted = snippet
    for w in words:
        if len(w) > 2:
            pattern = re.compile(re.escape(w), re.IGNORECASE)
            # Use RGBA for transparency (0.3 = 30% opacity)
            highlighted = pattern.sub(
                lambda
                    m: f'<span style="background-color: rgba(255, 215, 0, 0.3); color: inherit; padding: 0px 2px; border-radius: 4px;">{m.group(0)}</span>',
                highlighted
            )

    return highlighted


# --- MAIN UI FUNCTION ---
def render_search_ui(selected_video_name, video_path, video_player_placeholder, username, api_key):
    st.markdown("### 💬 Chat with Video")

    # 1. Initialize State
    if 'video_chat_history' not in st.session_state:
        st.session_state['video_chat_history'] = []

    # Check for video switch
    if 'last_video_name' not in st.session_state:
        st.session_state['last_video_name'] = selected_video_name
    elif st.session_state['last_video_name'] != selected_video_name:
        st.session_state['video_chat_history'] = []
        st.session_state['last_video_name'] = selected_video_name

    # Lock state for this specific component
    if 'processing_video' not in st.session_state:
        st.session_state['processing_video'] = False

    # 2. Scrollable Container
    chat_container = st.container(height=500)
    with chat_container:
        for i, msg in enumerate(st.session_state['video_chat_history']):
            with st.chat_message(msg['role'], avatar="🤖" if msg['role'] == "assistant" else None):
                st.write(msg['content'])
                if msg.get('sources'):
                    st.divider()
                    st.caption(f"Top {len(msg['sources'])} most relevant moments:")

                    for idx, res in enumerate(msg['sources']):
                        # Create a card for each result
                        with st.container(border=True):

                            # 1. BUTTON ROW (Top)
                            time_str = f"{int(res['start_time'] // 60):02d}:{int(res['start_time'] % 60):02d}"

                            # 'use_container_width=True' makes the button stretch to fill the width (optional)
                            if st.button(f"▶ Jump to {time_str}", key=f"vhist_{i}_{idx}", use_container_width=False):
                                st.session_state['start_time'] = res['start_time']
                                st.rerun()

                            # 2. TEXT ROW (Bottom)
                            # We use unsafe_allow_html=True so our yellow marker works
                            st.markdown(
                                f"<div style='margin-top: 10px; color: #CCCCCC;'>... \"{res['text']}\" ...</div>",
                                unsafe_allow_html=True)

    # 3. Locked Chat Input
    query = st.chat_input(
        "Ask about this video...",
        on_submit=lock_video_chat,
        disabled=st.session_state['processing_video']
    )

    if query and selected_video_name:
        st.session_state['video_chat_history'].append({"role": "user", "content": query})

        col_name = video_processor.get_safe_collection_name(selected_video_name)
        results = search_single_video(col_name, query, username)

        if results and results['documents']:
            found_any = True
            valid_results = []
            for i in range(len(results['documents'][0])):
                doc_text = results['documents'][0][i]
                start_time = results['metadatas'][0][i]['start_time']

                # This function now returns a SHORT snippet with FADED highlights
                styled_text = highlight_text(doc_text, query)

                valid_results.append({'text': styled_text, 'start_time': start_time})

            if found_any:
                with st.spinner("Analyzing..."):
                    ai_answer = ask_gemini(query, valid_results, api_key)
                    st.session_state['video_chat_history'].append({
                        "role": "assistant",
                        "content": ai_answer,
                        "sources": valid_results
                    })
        else:
            st.session_state['video_chat_history'].append({
                "role": "assistant",
                "content": "No matches found.",
                "sources": []
            })

        # UNLOCK AND RERUN
        st.session_state['processing_video'] = False
        st.rerun()
