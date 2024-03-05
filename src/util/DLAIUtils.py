import os
import sys
from dotenv import load_dotenv, find_dotenv
from pinecone import Pinecone

class Utils:
  def __init__(self):
    self.index_name = self.create_dlai_index_name('dl-ai');
    self.pinecone = Pinecone(api_key=self.get_pinecone_api_key())
    self.index = self.pinecone.Index(self.index_name)


  def create_dlai_index_name(self, index_name):
    openai_key = ''
    if self.is_colab(): # google colab
      from google.colab import userdata
      openai_key = userdata.get("PINECONE_API_KEY")
    else: # jupyter notebook
      openai_key = os.getenv("PINECONE_API_KEY")
    return f'{index_name}-{openai_key[-36:].lower().replace("_", "-")}'
    
  def is_colab(self):
    return 'google.colab' in sys.modules
  
  def set_index(self, index):
    self.index = index
  
  def get_index(self):
    return self.index
  
  def get_index_name(self):
    return self.index_name
  
  def get_openai_api_key(self):
    _ = load_dotenv(find_dotenv())
    return os.getenv("OPENAI_API_KEY")
    
  def get_pinecone_api_key(self):
    _ = load_dotenv(find_dotenv())
    return os.getenv("PINECONE_API_KEY")
  
  def get_pinecone(self):
    return self.pinecone