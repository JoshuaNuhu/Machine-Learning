import os
import glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import util
from sentence_transformers import SentenceTransformer

# This line suppresses the Hugging Face warning to keep your terminal clean
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_documents(data_dir="data"):

    documents = []
    file_pattern = os.path.join(data_dir, "*.txt")
    file_paths = glob.glob(file_pattern)
    
    if not file_paths:
        print(f"⚠️ Warning: No .txt files found in the '{data_dir}' directory.")
        return documents

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                text_content = file.read()
                if text_content.strip():
                    documents.append(text_content)
                    print(f"✅ Successfully loaded: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            
    print(f"--- Total documents loaded: {len(documents)} ---")
    return documents

def chunk_text(documents):

    all_chunks = []
    for doc in documents:
        # Split by double newline and remove empty strings
        chunks = [chunk.strip() for chunk in doc.split('\n\n') if chunk.strip()]
        all_chunks.extend(chunks)
        
    print(f"✂️  Chunking complete: Created {len(all_chunks)} total chunks.")
    return all_chunks

def create_embeddings(chunks, model_name='all-MiniLM-L6-v2'):
  
    print(f"🧠 Step 1: Connecting to Hugging Face to load '{model_name}'...")
    # This line is where the 'lock' usually happens if the cache is corrupted
    model = SentenceTransformer(model_name)
    
    print(f"🧪 Step 2: Model loaded! Encoding {len(chunks)} chunks...")
    print("⏳ NOTE: This will take ~2-5 minutes. Please do not interrupt.")
    
    # This part does the heavy math
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    print("✅ Step 3: Embeddings created successfully!")
    return model, np.array(embeddings)
def search_documents(query, model, embeddings, chunks, top_k=3):

    # Step 1: Encode the user's question
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Step 2: Calculate Cosine Similarity
    # This compares our 1 query against all 7,000+ chunks at once!
    cosine_scores = util.cos_sim(query_embedding, embeddings)[0]
    
    # Step 3: Find the indices of the highest scores
    # topk returns the values and indices of the best matches
    top_results = util.semantic_search(query_embedding, embeddings, top_k=top_k)
    
    # Step 4: Format the results
    results = []
    for hit in top_results[0]:
        idx = hit['corpus_id']
        score = hit['score']
        results.append({
            "text": chunks[idx],
            "score": round(float(score), 4)
        })
        
    return results
def keyword_search(query, chunks, top_k=3):
    """
    Traditional search based only on exact word matches (TF-IDF).
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    # Transform all chunks into keyword counts
    tfidf_matrix = vectorizer.fit_transform(chunks)
    # Transform the query
    query_vec = vectorizer.transform([query])
    
    # Calculate similarity
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get top indices
    top_indices = scores.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            "text": chunks[idx],
            "score": round(float(scores[idx]), 4)
        })
    return results
# --- FULL TEST SEQUENCE ---
if __name__ == "__main__":
    print("--- Finalizing Search Logic ---")
    raw_docs = load_documents()
    
    if raw_docs:
        final_chunks = chunk_text(raw_docs)
        model, embeddings = create_embeddings(final_chunks)
        
        # --- TEST SEARCH ---
        print("\n" + "="*30)
        user_query = "What is the definition of machine learning?" # Change this to a question relevant to your notes!
        print(f"Searching for: '{user_query}'")
        
        matches = search_documents(user_query, model, embeddings, final_chunks)
        
        print("\nTOP RESULTS:")
        for i, res in enumerate(matches):
            print(f"\n[{i+1}] (Similarity Score: {res['score']})")
            print(f"Content: {res['text'][:200]}...") # Print first 200 chars
        print("="*30)