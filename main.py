from flask import Flask, render_template, request
import argparse
from src.models.predict_model import PredictionModel
from src.models.train_model import TrainModel
from src.util.DLAIUtils import Utils
from flask import jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/semantic-search', methods=['POST'])
def semantic_search():
    query = request.json['query']
    predictmodel = PredictionModel()
    context = predictmodel.search_relevant_text(query)
    return predictmodel.generate_prompt_openai(query,context)

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/websites', methods=['POST'])
def upload_website_data():
    urls = request.json['urls']
    trainmodel = TrainModel()
    trainmodel.upload_website_data_to_pinecone(urls, 'dl-ai')
    return "successfully uploaded data to pinecone"

@app.route('/files', methods=['POST'])
def upload_file_data():
    files = request.files.getlist('file[]')
    trainmodel = TrainModel()
    trainmodel.save_files(files)
    trainmodel.upload_file_content_to_pinecone('dl-ai')
    return "successfully uploaded data to pinecone"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Semantic Search')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    args = parser.parse_args()
    app.run(port=args.port)
