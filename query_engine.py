import streamlit as st
import video_processor
from sentence_transformers import CrossEncoder
import google.generativeai as genai

# --- CONFIGURATION ---
INITIAL_TOP_K = 10
FINAL_TOP_K = 3
CONFIDENCE_THRESHOLD = 0.45


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
        col_name = "".join([c if c.isalnum() else "_" for c in video_name])
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

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel('gemini-2.5-flash')

    context_text = ""
    for item in context_results:
        context_text += f"- From video '{item['video_name']}': {item['text']}\n"

    prompt = f"""
    You are a helpful video assistant. 
    Answer the user's question based ONLY on the context provided below.
    If the answer is not in the context, say "I couldn't find the answer in the videos."

    Context:
    {context_text}

    User Question: {query}

    Answer:
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error connecting to Gemini: {e}"


def render_search_ui(selected_video_name, video_path, video_player_placeholder, username):
    """UI לחיפוש בתוך וידאו ספציפי"""
    st.subheader("Deep Search in Video")
    query = st.text_input("Find specific moment in this video...", key="local_search")

    if query and selected_video_name:
        col_name = "".join([c if c.isalnum() else "_" for c in selected_video_name])

        # כעת אנו מעבירים גם את שם המשתמש
        results = search_single_video(col_name, query, username)

        if results and results['documents']:
            found_any = False
            for i in range(len(results['documents'][0])):
                score = results['distances'][0][i]

                # סינון תוצאות לא רלוונטיות
                if score > CONFIDENCE_THRESHOLD:
                    continue

                found_any = True
                doc_text = results['documents'][0][i]
                start_time = results['metadatas'][0][i]['start_time']
                time_str = f"{int(start_time // 60):02d}:{int(start_time % 60):02d}"

                with st.expander(f"{time_str} - {doc_text[:50]}..."):
                    st.write(f"\"{doc_text}\"")
                    if st.button(f"Jump to {time_str}", key=f"jump_{i}"):
                        video_player_placeholder.video(video_path, start_time=int(start_time))

            if not found_any:
                st.warning("No relevant matches found in this video.")
