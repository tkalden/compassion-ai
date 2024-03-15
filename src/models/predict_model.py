from openai import OpenAI
from sentence_transformers import SentenceTransformer
import torch
from src.util.DLAIUtils import Utils
import logging


logging.basicConfig(level=logging.INFO)

class PredictionModel:
    def __init__(self):
        self.utils = Utils()
        self.openai_client = OpenAI(api_key=self.utils.get_openai_api_key())
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
        self.pinecone = self.utils.get_pinecone()
        self.index = self.utils.get_index()
        self.context_limt=10000


    def search_relevant_text(self, query: str):
        logging.info('Searching for relevant text for: %s', query)     
        results = self.index.query(top_k=10, vector=self.model.encode(query).tolist(), include_metadata=True, include_values=False)
        top_results = []
        for result in results['matches']:
             if(result['score'] > 0.5 and len(top_results) < 3): #top 3 results
                top_results.append(result['metadata'])
                logging.info('Score: %s | Result: %s', result['score'], result['metadata'])
                logging.info('='*100)
        return top_results
   
    def generate_prompt_openai(self, query: str, relevant_text: list):
         # get relevant contexts
        contexts = [item['text'] for item in relevant_text]   
        if contexts == []:
            return {'answer':"No Relevant Information Found. Please upload some more information to know the answer to your question."}  
        # build our prompt with the retrieved contexts included
        prompt_start = (
            "Answer the question based on the context below.\n\n"+
            "Context:\n"
        )
        prompt_end = (
            f"\n\nQuestion: {query}\nAnswer:"
        )
       
        prompt = (prompt_start +' '.join(' '.join(contexts).split()[:self.context_limt]) + prompt_end)

        res = self.openai_client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            temperature=0,
            max_tokens=636,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        logging.info('OpenAI response: %s', res.choices[0])
        sources = [item['source'] for item in relevant_text]    
        return {'answer': res.choices[0].text,'sources': list(set(sources))}