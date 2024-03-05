# Semantic Search Using Pincone Vector Database and SentenceTransformer 
In this project, you need to pass urls of websites and indexname of vector database. train_model.py will scrapp off data from given urls, parse sentences from data and convert each sentence to vector using Sentence Transformer. Each url gets n vecotrs. 
```The vector_id for each url is  DLA + subdomainname + last path variable + i  where i is the index of ith vector. ```

```
http://example1.com/test gets n vector_ids -> [DLAexample1test1,DLAexample1test2,DLAexample1test3,....DLAexample1testn]
``` 
Data is inserted into vector database in batch of 100 vectors. Then semantic search is performed against the data stored in the vector database.

## Environment Setup
### using conda
    conda -V
    conda update conda
    conda create -n envname python=x.x anaconda
    conda activate envname
    pip install -r  requirements.txt
    
I recommend setting up environment using conda because most of the ai related packages are alreay installed 
For more information - https://www.geeksforgeeks.org/set-up-virtual-environment-for-python-using-anaconda/

### using venv
    python3 --version in your terminal  
    python3 -m venv venv
    source venv/bin/activate
    pip install -r  requirements.txt

## Pinecone API Key 
   Follow the instruction here (https://docs.pinecone.io/docs/quickstart). Once you have the api key, write it on the .env file for e.g. PINECONE_API_KEY = f4c9c474-73e7-4a4e-9dae-4c96a4bb084a

## Running the app using Flask 
    python main.py

Hit the endpoint on 127.0.0.1:5000

## Running the app from terminal using the arguments
### upload data 
    python upload_data.py --index_name "my-index" --urls "https://spaceplace.nasa.gov/blue-sky/en/" "https://www.britannica.com/science/sky" 
Note the index_name is the index of the vector database where you will store all the information. Name must consist of lower case alphanumeric characters or '-'.
### query data
    python query_data.py --query "why the sky is blue?"

## Running the app using flask

    python main.py



