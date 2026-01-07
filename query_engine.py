import streamlit as st
import video_processor


def search_single_video(collection_name, query_text, n_results=3):
    """Searches within a specific video (used after you select one)."""
    client = video_processor.get_db_client()
    try:
        collection = client.get_collection(collection_name)
        results = collection.query(query_texts=[query_text], n_results=n_results)
        return results
    except ValueError:
        return None


def search_all_collections(query_text, top_k=3):
    """
    Scans ALL videos to find the best 3 matches across the entire library.
    Returns a sorted list of matches.
    """
    client = video_processor.get_db_client()
    videos = video_processor.get_videos_list()

    all_candidates = []

    # 1. Iterate over every video in the library
    for video_name in videos:
        # Reconstruct the valid collection name
        col_name = "".join([c if c.isalnum() else "_" for c in video_name])

        try:
            collection = client.get_collection(col_name)
            # Get the single best match from this video
            results = collection.query(query_texts=[query_text], n_results=1)

            if results['documents'] and results['documents'][0]:
                # Extract the data
                doc_text = results['documents'][0][0]
                meta = results['metadatas'][0][0]
                score = results['distances'][0][0]  # Lower distance = better match

                all_candidates.append({
                    "video_name": video_name,
                    "reason": doc_text,  # The sentence that triggered the match
                    "start_time": meta['start_time'],
                    "score": score
                })
        except Exception:
            # Skip if collection doesn't exist or error
            continue

    # 2. Sort all candidates by score (Lower distance is better)
    all_candidates.sort(key=lambda x: x['score'])

    # 3. Return only the top K global results
    return all_candidates[:top_k]


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
