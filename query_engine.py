import streamlit as st
import video_processor  # To access the shared DB client


def search_collection(collection_name, query_text, n_results=3):
    client = video_processor.get_db_client()
    try:
        collection = client.get_collection(collection_name)
        results = collection.query(query_texts=[query_text], n_results=n_results)
        return results
    except ValueError:
        # Collection might not exist yet (processing)
        return None


def render_search_ui(selected_video_name, video_path, video_player_placeholder):
    st.subheader("Search")
    query = st.text_input("Ask a question about this video...")

    if query and selected_video_name:
        # Generate the collection name using the same logic as ingestion
        collection_name = "".join([c if c.isalnum() else "_" for c in selected_video_name])

        results = search_collection(collection_name, query)

        if results and results['documents']:
            st.write("Results:")
            for i in range(len(results['documents'][0])):
                doc_text = results['documents'][0][i]
                meta = results['metadatas'][0][i]
                start_time = meta['start_time']

                time_str = f"{int(start_time // 60):02d}:{int(start_time % 60):02d}"

                with st.expander(f"{time_str} - Match"):
                    st.write(f"\"{doc_text}\"")
                    # Unique key for every button
                    if st.button(f"Jump to {time_str}", key=f"btn_{selected_video_name}_{i}"):
                        video_player_placeholder.video(video_path, start_time=int(start_time))
        else:
            st.warning("No results found or video is still processing.")
