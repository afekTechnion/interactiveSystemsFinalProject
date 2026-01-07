import streamlit as st
import video_processor
import streamlit as st
import video_processor
from sentence_transformers import CrossEncoder

# --- CONFIGURATION ---
# We fetch more results initially (10) to let the Reranker filter them down
INITIAL_TOP_K = 10
FINAL_TOP_K = 3


def search_single_video(collection_name, query_text, n_results=3):
    """Searches within a specific video (used after you select one)."""
    client = video_processor.get_db_client()
    try:
        collection = client.get_collection(collection_name)
        results = collection.query(query_texts=[query_text], n_results=n_results)
        return results
    except ValueError:
        return None


# Load the Cross-Encoder (The "Judge")
# 'ms-marco' is a model specifically trained to match Questions to Answers
@st.cache_resource
def load_reranker():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def search_all_collections(query_text):
    """
    Reranked Global Search:
    1. Retrieve top 10 matches using fast Vector Search.
    2. Rerank them using a Cross-Encoder (Smart Q&A matching).
    3. Return the top 3 best answers.
    """
    client = video_processor.get_db_client()
    videos = video_processor.get_videos_list()
    reranker = load_reranker()

    # --- STAGE 1: Retrieval (Fast but "Blurry") ---
    initial_candidates = []

    for video_name in videos:
        col_name = "".join([c if c.isalnum() else "_" for c in video_name])
        try:
            collection = client.get_collection(col_name)
            # Fetch top 2 from EVERY video to get a wide pool of candidates
            results = collection.query(query_texts=[query_text], n_results=2)

            if results['documents']:
                for i in range(len(results['documents'][0])):
                    doc_text = results['documents'][0][i]
                    meta = results['metadatas'][0][i]

                    initial_candidates.append({
                        "video_name": video_name,
                        "text": doc_text,
                        "start_time": meta['start_time']
                    })
        except Exception:
            continue

    if not initial_candidates:
        return []

    # --- STAGE 2: Reranking (Slow but "Smart") ---
    # Prepare pairs: [[Query, Text1], [Query, Text2], ...]
    rerank_pairs = [[query_text, candidate['text']] for candidate in initial_candidates]

    # The model gives a relevance score for each pair
    scores = reranker.predict(rerank_pairs)

    # Attach scores to candidates
    for i, candidate in enumerate(initial_candidates):
        candidate['score'] = scores[i]
        candidate['reason'] = candidate['text']  # For UI compatibility

    # Sort by the new Cross-Encoder score (Higher is better now!)
    # Note: Cross-Encoder scores are usually logits (can be negative), higher = more relevant
    initial_candidates.sort(key=lambda x: x['score'], reverse=True)

    # Return the top 3 survivors
    return initial_candidates[:FINAL_TOP_K]


def render_search_ui(selected_video_name, video_path, video_player_placeholder):
    """(Kept for local search inside the player)"""
    st.subheader("Deep Search in Video")
    query = st.text_input("Find specific moment in this video...", key="local_search")

    if query and selected_video_name:
        col_name = "".join([c if c.isalnum() else "_" for c in selected_video_name])
        results = search_single_video(col_name, query)

        if results and results['documents']:
            for i in range(len(results['documents'][0])):
                doc_text = results['documents'][0][i]
                start_time = results['metadatas'][0][i]['start_time']
                time_str = f"{int(start_time // 60):02d}:{int(start_time % 60):02d}"

                with st.expander(f"{time_str} - {doc_text[:50]}..."):
                    st.write(f"\"{doc_text}\"")
                    if st.button(f"Jump to {time_str}", key=f"jump_{i}"):
                        video_player_placeholder.video(video_path, start_time=int(start_time))
