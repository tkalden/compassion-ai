from ast import List
import warnings

from flask import jsonify

from src.util.DLAIUtils import Utils

warnings.filterwarnings('ignore')
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import torch
from tqdm.auto import tqdm
from PyPDF2 import PdfReader
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

utils = Utils()

PINECONE_API_KEY = utils.get_pinecone_api_key()

#need to check when uploading that same data is not inserting multiple times


class SemanticSearch:
    def __init__(self):
        self.index_name = utils.create_dlai_index_name('dl-ai')
        self.pinecone = Pinecone(api_key=PINECONE_API_KEY)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device != 'cuda':
            print('Sorry no cuda.')
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.batch_size = 200
        self.text_length_limit = 500
        self.vector_limit = 10000
        self.combined_text = []
        self.index = self.pinecone.Index(self.index_name)
 
    def upload_data_to_pinecone(self, files, urls, index_name):
        self.index_name = utils.create_dlai_index_name(index_name)
        self.create_new_index(self.index_name)
        #need to work on parsing the file name
        for file in files:
            self.combined_text.extend([(self.parse_url(file) + str(i), v) for i, v in enumerate(self.extract_text_from_pdf(file))])
        for url in urls:
            self.combined_text.extend([(self.parse_url(url) + str(i), v) for i, v in enumerate(self.extract_text_from_url(url))])
        dict_array = dict(self.combined_text)
        questions = list(dict_array.values())[:self.vector_limit]
        print('Total number of text to be uploaded:', len(questions))
        for i in tqdm(range(0, len(questions), self.batch_size)):
            print('Uploading batch:', i, 'to', i+self.batch_size)
            # find end of batch
            i_end = min(i+self.batch_size, len(questions))
            # create IDs batch
            ids = list(dict_array.keys())[i:i_end]
            print("IDs of vectors", ids)
            # create metadata batch
            metadatas = [{'text': text[:self.text_length_limit]} for text in questions[i:i_end]]
            # create embeddings
            xc = self.model.encode(questions[i:i_end])
            # create records list for upsert
            records = zip(ids, xc, metadatas)
            # upsert to Pinecone
            self.index.upsert(vectors=records)

    def create_new_index(self, index_name):
        if index_name not in [index.name for index in self.pinecone.list_indexes()]:
            print('Creating index:', index_name)
            self.pinecone.create_index(name=index_name, 
            dimension=self.model.get_sentence_embedding_dimension(), 
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-west-2')) 
        
    def parse_url(self,url):
        parsed_url = urlparse(url)
        print('Parsed URL:', parsed_url)
        domain = parsed_url.netloc.split('.')[-2]
        last_four_letters = parsed_url.path.split('/')[-1]
        url_index = domain + last_four_letters
        print('URL index:', url_index)
        return 'DLA' + url_index
    

    def search_relevant_text(self, query: str):
        #defaut index name
        if self.index is None:  
            self.index = self.pinecone.index(self.utils.create_dlai_index_name('dl-ai'))
        print('Searching for relevant text for:', query)
        embedding = self.model.encode(query).tolist()
        results = self.index.query(top_k=10, vector=embedding, include_metadata=True, include_values=False)
        relevant_text = {}
        for result in reversed(results['matches']):
            score = round(result['score'], 2)
            text = result['metadata']['text']
            relevant_text[score] = text
        print('Relevant text:', relevant_text)
        return relevant_text

    def extract_text_from_pdf(self,file):
        pdf = PdfReader(file)
        text = " ".join(page.extract_text() for page in pdf.pages)       
        return sent_tokenize(text)
    
    def extract_text_from_url(self,url):
        print('Extracting text from:', url)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find all paragraph 'p' tags and extract the text
        text = ' '.join(p.get_text() for p in soup.find_all('p'))
        return sent_tokenize(text)