import streamlit as st
import video_processor
from sentence_transformers import CrossEncoder
import google.generativeai as genai

# --- CONFIGURATION ---
INITIAL_TOP_K = 10
FINAL_TOP_K = 3


@st.cache_resource
def load_reranker():
    # Loads the Cross-Encoder model to judge relevance
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def search_all_collections(query_text):
    """
    Retrieves and Reranks results to find the best context.
    """
    client = video_processor.get_db_client()
    videos = video_processor.get_videos_list()
    reranker = load_reranker()

    # 1. Broad Search (Retrieval)
    initial_candidates = []
    for video_name in videos:
        col_name = "".join([c if c.isalnum() else "_" for c in video_name])
        try:
            collection = client.get_collection(col_name)
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

    # 2. Smart Filtering (Reranking)
    rerank_pairs = [[query_text, candidate['text']] for candidate in initial_candidates]
    scores = reranker.predict(rerank_pairs)

    for i, candidate in enumerate(initial_candidates):
        candidate['score'] = scores[i]
        candidate['reason'] = candidate['text']

    initial_candidates.sort(key=lambda x: x['score'], reverse=True)
    return initial_candidates[:FINAL_TOP_K]


def ask_gemini(query, context_results, api_key):
    """
    The 'G' in RAG: Sends the context + question to Gemini.
    """
    if not api_key:
        return "Please enter your Google API Key in the sidebar to generate an answer."

    # Configure the API with the user's key
    genai.configure(api_key=api_key)

    # Use Gemini Pro (optimized for text)
    model = genai.GenerativeModel('gemini-1.5-flash')

    # Construct the Prompt
    # We feed the retrieved video snippets as "Context"
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


# ... (Include the render_search_ui function from previous steps here if needed) ...
def render_search_ui(selected_video_name, video_path, video_player_placeholder):
    # Same as previous version
    st.subheader("Deep Search in Video")
    query = st.text_input("Find specific moment in this video...", key="local_search")

    if query and selected_video_name:
        col_name = "".join([c if c.isalnum() else "_" for c in selected_video_name])
        # Note: You need the search_single_video function here too (omitted for brevity)
        # You can copy it from the previous response
        pass