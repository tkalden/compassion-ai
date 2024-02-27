from flask import Flask, render_template, request
import argparse
from src.models.train_model import SemanticSearch
from src.util.DLAIUtils import Utils
from flask import jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/semantic-search', methods=['POST'])
def semantic_search():
    query = request.json['query']
    semantic_search = SemanticSearch()
    results = semantic_search.search_relevant_text(query)
    filtered_results = {key: value for key, value in results.items() if key > 0.5}
    return filtered_results

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/upload-data', methods=['POST'])
def upload_data():
    index_name = request.json['index_name']
    urls = request.json['urls']
    semantic_search = SemanticSearch()
    semantic_search.upload_data_to_pinecone([], urls,index_name)
    return "successfully uploaded data to pinecone"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Semantic Search')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    args = parser.parse_args()
    app.run(port=args.port)
