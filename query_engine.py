import streamlit as st
import video_processor
import torch
from sentence_transformers import CrossEncoder
import google.generativeai as genai

# --- CONFIGURATION ---
GEMINI_MODEL_NAME = "gemini-2.5-flash"  # המודל הראשי
INITIAL_TOP_K = 10
FINAL_TOP_K = 3


@st.cache_resource(show_spinner=False)
def load_reranker():
    # --- GPU VERIFICATION ---
    if torch.cuda.is_available():
        device = "cuda"
        print("\n\n✅✅✅ GPU DETECTED: RUNNING IN FAST MODE ✅✅✅\n\n")
    else:
        device = "cpu"
        print("\n\n⚠️⚠️⚠️ GPU NOT FOUND: RUNNING IN SLOW CPU MODE ⚠️⚠️⚠️\n\n")

    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)


def expand_context(collection, center_id, window=1):
    try:
        base_name, index_str = center_id.rsplit('_', 1)
        current_idx = int(index_str)

        ids_to_fetch = []
        for i in range(current_idx - window, current_idx + window + 1):
            if i >= 0:
                ids_to_fetch.append(f"{base_name}_{i}")

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
        results = collection.query(query_texts=[query_text], n_results=n_results)
        return results
    except ValueError:
        return None


def search_all_collections(query_text, username):
    videos_dir, chroma_dir, _ = video_processor.get_user_paths(username)
    client = video_processor.get_db_client(chroma_path=chroma_dir)
    videos = video_processor.get_videos_list(username)

    reranker = load_reranker()

    initial_candidates = []
    # Loop - No Parallel Search yet (as requested)
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

    if not initial_candidates:
        return []

    # Re-ranking
    rerank_pairs = [[query_text, candidate['text']] for candidate in initial_candidates]
    scores = reranker.predict(rerank_pairs)

    for i, candidate in enumerate(initial_candidates):
        candidate['score'] = scores[i]
        candidate['reason'] = candidate['text']

    initial_candidates.sort(key=lambda x: x['score'], reverse=True)
    return initial_candidates[:FINAL_TOP_K]


# --- NEW: LOCAL FALLBACK LOGIC ---
def format_local_fallback(query, context_results, error_msg):
    """
    This runs when Gemini fails. It formats the raw data nicely.
    This GUARANTEES an answer even without internet.
    """
    response = f"**⚠️ Cloud AI Unavailable ({error_msg})**\n\n"
    response += "Using **Local Fallback Mode**. Here is the most relevant information found in your video library:\n\n"

    for i, item in enumerate(context_results):
        response += f"**{i + 1}. From video: {item.get('video_name', 'Unknown')}**\n"
        response += f"> *\"{item['text']}\"*\n\n"

    response += "\n*These are the direct transcript matches.*"
    return response


def ask_gemini(query, context_results, api_key):
    """
    Tries Gemini 2.5 -> Falls back to Gemini Pro -> Falls back to Local Raw Text
    """
    # 1. Check Key
    if not api_key:
        return format_local_fallback(query, context_results, "No API Key provided")

    try:
        genai.configure(api_key=api_key)

        context_text = ""
        for item in context_results:
            context_text += f"- {item['text']}\n"

        prompt = f"""
        You are a helpful video assistant. 
        Answer the user's question based ONLY on the context provided below.

        Context:
        {context_text}

        User Question: {query}

        Answer:
        """

        # 2. Try Primary Model (Gemini 2.5 Flash)
        try:
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            response = model.generate_content(prompt)
            return response.text

        except Exception as e_primary:
            print(f"⚠️ Primary model {GEMINI_MODEL_NAME} failed: {e_primary}")

            # 3. Try Secondary Model (Gemini 1.5 Pro - standard fallback)
            try:
                print("🔄 Attempting fallback to gemini-pro...")
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt)
                return response.text

            except Exception as e_secondary:
                # 4. FINAL FALLBACK: Local Mode (Works 100% of the time)
                print(f"❌ All Cloud models failed. Switching to Local Mode.")
                return format_local_fallback(query, context_results, "Connection Error")

    except Exception as e_global:
        return format_local_fallback(query, context_results, str(e_global))


def render_search_ui(selected_video_name, video_path, video_player_placeholder, username, api_key):
    st.markdown("### 💬 Chat with Video")
    query = st.chat_input("Ask about this video...")

    if 'local_answer' in st.session_state and st.session_state.get('local_vid') == selected_video_name:
        with st.chat_message("assistant", avatar="🤖"):
            st.write(st.session_state['local_answer'])

            # Show "Found at" only if it's NOT a fallback error message (optional check)
            st.caption("📍 **Found at:**")
            results = st.session_state.get('local_results', [])
            for res in results:
                time_str = f"{int(res['start_time'] // 60):02d}:{int(res['start_time'] % 60):02d}"
                if st.button(f"▶ Play at {time_str}", key=f"jump_{res['start_time']}", use_container_width=True):
                    video_player_placeholder.video(video_path, start_time=int(res['start_time']))

    if query and selected_video_name:
        st.session_state['local_vid'] = selected_video_name
        with st.chat_message("user"):
            st.write(query)

        col_name = video_processor.get_safe_collection_name(selected_video_name)
        results = search_single_video(col_name, query, username)

        if results and results['documents']:
            found_any = True
            valid_results = []
            for i in range(len(results['documents'][0])):
                doc_text = results['documents'][0][i]
                start_time = results['metadatas'][0][i]['start_time']
                valid_results.append({'text': doc_text, 'start_time': start_time})

            if found_any:
                # שינוי קטן בטקסט הספינר כדי שייראה מקצועי
                with st.spinner("Analyzing..."):
                    ai_answer = ask_gemini(query, valid_results, api_key)
                    st.session_state['local_answer'] = ai_answer
                    st.session_state['local_results'] = valid_results
                    st.rerun()
            else:
                st.warning("No matches found.")