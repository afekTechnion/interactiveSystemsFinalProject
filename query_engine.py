import streamlit as st
import video_processor
from sentence_transformers import CrossEncoder
from google import genai

# --- CONFIGURATION ---
INITIAL_TOP_K = 10
FINAL_TOP_K = 3
CONFIDENCE_THRESHOLD = 1


@st.cache_resource
def load_reranker():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


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
    """מחפש בתוך וידאו ספציפי של משתמש ספציפי"""
    _, chroma_dir = video_processor.get_user_paths(username)
    client = video_processor.get_db_client(chroma_dir)
    try:
        collection = client.get_collection(collection_name)
        results = collection.query(query_texts=[query_text], n_results=n_results)
        return results
    except ValueError:
        return None


def search_all_collections(query_text, username):
    videos_dir, chroma_dir = video_processor.get_user_paths(username)
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

    if not initial_candidates:
        return []

    rerank_pairs = [[query_text, candidate['text']] for candidate in initial_candidates]
    scores = reranker.predict(rerank_pairs)

    for i, candidate in enumerate(initial_candidates):
        candidate['score'] = scores[i]
        candidate['reason'] = candidate['text']

    initial_candidates.sort(key=lambda x: x['score'], reverse=True)
    return initial_candidates[:FINAL_TOP_K]


def ask_gemini(query, context_results, api_key):
    if not api_key:
        return "Please enter your Google API Key in the sidebar to generate an answer."

    # --- NEW SYNTAX (google-genai) ---
    try:
        # 1. Initialize the Client
        client = genai.Client(api_key=api_key)

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

        # 2. Generate Content using the Client
        # Note: I changed the model to 'gemini-1.5-flash' because '2.5' does not exist yet.
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text

    except Exception as e:
        return f"Error connecting to Gemini: {e}"


def render_search_ui(selected_video_name, video_path, video_player_placeholder, username, api_key):
    st.subheader("Deep Search in Video")
    query = st.text_input("Find specific moment in this video...", key="local_search")

    if query and selected_video_name:
        # Use the safe name converter
        col_name = video_processor.get_safe_collection_name(selected_video_name)

        # Pass the username
        results = search_single_video(col_name, query, username)

        if results and results['documents']:
            found_any = False
            valid_results = []

            # 1. Filter results
            for i in range(len(results['documents'][0])):
                score = results['distances'][0][i]
                if score <= CONFIDENCE_THRESHOLD:
                    found_any = True
                    doc_text = results['documents'][0][i]
                    start_time = results['metadatas'][0][i]['start_time']
                    valid_results.append({'text': doc_text, 'start_time': start_time})

            if found_any:
                # 2. GENERATE ANSWER (This uses the api_key you passed)
                with st.spinner("Generating AI Answer..."):
                    ai_answer = ask_gemini(query, valid_results, api_key)
                    st.markdown(f"**🤖 AI Answer:** {ai_answer}")
                    st.divider()

                # 3. Show Timestamps
                for res in valid_results:
                    time_str = f"{int(res['start_time'] // 60):02d}:{int(res['start_time'] % 60):02d}"
                    with st.expander(f"Jump to {time_str}"):
                        st.write(f"\"{res['text']}\"")
                        if st.button(f"Play {time_str}", key=f"jump_{res['start_time']}"):
                            video_player_placeholder.video(video_path, start_time=int(res['start_time']))
            else:
                st.warning("No matches found.")
