import warnings
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import torch
from PyPDF2 import PdfReader
import requests
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from werkzeug.utils import secure_filename
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from src.util.DLAIUtils import Utils
import os
from pinecone import ServerlessSpec
import logging

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')


class TrainModel:
    def __init__(self):
        self.current_directory = os.getcwd()
        self.upload_folder = os.path.join(self.current_directory, 'data/raw')
        self.allowed_files = {'pdf'}
        self.utils = Utils()
        self.pinecone = self.utils.get_pinecone()
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 200
        self.text_length_limit = 500
        self.vector_limit = 10000
        self.website_content = []
        self.file_content = []
        self.uploaded_file_names = []
        self.tokenizer = tiktoken.get_encoding('cl100k_base')


    def upload_website_data_to_pinecone(self, urls, index_name):
        for content in [self.extract_text_from_url(url) for url in urls]:
            self.upsert_data_to_pinecone(content,index_name,True)
    
    def upsert_data_to_pinecone(self, data,index_name, is_website=False):
        logging.info('Upserting data %s to pinecone',data)
        self.index_name = self.utils.create_dlai_index_name(index_name)
        self.utils.set_index(self.create_new_index(self.index_name))
        # first get metadata fields for this record
        metadata = {
            'source': data['url'],
            'title': data['title']
        }
        primary_key_prefix =  self.create_primary_key_from_url(data['url']) if is_website else self.create_primary_key_from_file(data['url'])
        
        splitted_text = self.text_splitter(data['text'])
        # create individual metadata dicts for each chunk
        metadatas = [{
            "chunk": j, "text": text, **metadata
        } for j, text in enumerate(splitted_text)]
        logging.info('Total number of texts to be uploaded: %s', len(splitted_text))
        for i in tqdm(range(0, len(splitted_text), self.batch_size)):
            logging.info('Uploading batch: %s to %s', i,  i+self.batch_size)
            i_end = min(i+self.batch_size, len(splitted_text))
            ids = [primary_key_prefix + str(i) for i in range(i, i_end)]
            metadatas = [metadata for metadata in metadatas[i:i_end]]
            xc = self.model.encode(splitted_text[i:i_end])
            records = zip(ids, xc, metadatas)
            self.utils.get_index().upsert(vectors=records)

    
    def upload_file_content_to_pinecone(self, index_name):
        for content in [self.extract_text_from_pdf(file) for file in self.uploaded_file_names]:
            self.upsert_data_to_pinecone(content,index_name)

    def create_new_index(self, index_name):
        if self.utils.get_index_name() not in [index.name for index in self.pinecone.list_indexes()]:
            logging.info('Creating index: %s', index_name)
            self.pinecone.create_index(
                name=index_name,
                dimension=self.model.get_sentence_embedding_dimension(),
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-west-2')
            )
        return self.pinecone.Index(self.index_name)

    def create_primary_key_from_url(self, url):
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.split('.')[-2]
        last_four_letters = parsed_url.path.split('/')[-1]
        url_index = domain + last_four_letters
        return 'DLA' + url_index
    
    def create_primary_key_from_file(self, file):
        return 'DLA' + file.split('.pdf')[0]

    def extract_text_from_pdf(self, file):
        logging.info('Extracting text from: %s', file)
        pdf = PdfReader(os.path.join(self.upload_folder, file))
        text = " ".join(page.extract_text() for page in pdf.pages)
        text_with_source = {'url': file, 'text': text, 'title': file}
        return text_with_source

    def extract_text_from_url(self, url):
        logging.info('Extracting text from: %s', url)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join(p.get_text() for p in soup.find_all('p'))
        text_with_source = {'url': url, 'text': text, 'title': soup.title.string}
        return text_with_source
    
    def save_files(self, files):
        logging.info(files)
        for file in files:
            if file.filename == '':
                return "No selected file"
            if file and  not self.allowed_type(file.filename):
                return "Invalid file type"
            filename = secure_filename(file.filename)
            self.uploaded_file_names.append(filename)
            file.save(os.path.join(self.upload_folder, filename))
    
    def get_uploaded_file_names(self):
        return self.uploaded_file_names
    
    def allowed_type(self,filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.allowed_files
    
    #need to research more on chunk_size and chunk_overlap
    def text_splitter(self,text):
        return RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=20,
            length_function=self.tiktoken_len,
            separators=["\n\n", "\n", " ", ""]
        ).split_text(text)

    # create the length function
    def tiktoken_len(self,text):
        tokens = self.tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)
            