from flask import Flask, render_template, request
from utils import load_documents, chunk_text, create_embeddings, search_documents

app = Flask(__name__)

# --- Initialization ---
print("🚀 Initializing AI System... please wait.")
RAW_DOCS = load_documents()
CHUNKS = chunk_text(RAW_DOCS)
MODEL, EMBEDDINGS = create_embeddings(CHUNKS)
print("✅ AI System Ready!")

@app.route("/", methods=["GET", "POST"])
def index():
    # 1. Initialize all variables to None/False for the first page load
    ai_reply = None
    user_query = None
    is_error = False
    accuracy = None

    # 2. Handle the user submitting a question
    if request.method == "POST":
        user_query = request.form.get("query")
        
        if user_query:
            # Search for the top 2 matching chunks
            results = search_documents(user_query, MODEL, EMBEDDINGS, CHUNKS, top_k=2)
            
            # 3. AI Logic: Check confidence
            if not results or results[0]['score'] < 0.50:
                is_error = True
                ai_reply = "I'm sorry, but I have not been trained on that specific topic. I cannot find a reliable answer in your provided course materials."
                accuracy = None # No accuracy to show on an error
            else:
                # Calculate the percentage and format it (e.g., 0.854 -> "85.4%")
                best_score = results[0]['score']
                accuracy = f"{round(best_score * 100, 1)}%"
                
                # Extract and format the text
                extracted_text = "\n\n".join([res['text'] for res in results])
                ai_reply = f"Based on your course materials, here is the information I found:\n\n{extracted_text}"
                
    # 4. Pass EVERYTHING to the HTML template
    return render_template(
        "index.html", 
        ai_reply=ai_reply, 
        user_query=user_query, 
        is_error=is_error, 
        accuracy=accuracy
    )

if __name__ == "__main__":
    app.run(debug=True)