# Semantic Search Using Pincone Vector Database and SentenceTransformer 
In this project, you need to pass urls of websites and indexname of vector database. train_model.py will scrapp off data from given urls, parse sentences from data and convert each sentence to vector using Sentence Transformer. Each url gets n vecotrs. 
```The vector_id for each url is  DLA + subdomainname + last path variable + i  where i is the index of ith vector. ```

```
http://example1.com/test gets n vector_ids -> [DLAexample1test1,DLAexample1test2,DLAexample1test3,....DLAexample1testn]
``` 
Data is inserted into vector database in batch of 100 vectors. Then semantic search is performed against the data stored in the vector database.

## Environment Setup
### using conda
1. I recommend setting up environment using conda - https://www.geeksforgeeks.org/set-up-virtual-environment-for-python-using-anaconda/
### using venv
1. First, ensure that you have Python 3 installed on your system. You can check this by running
    ``` 
    python3 --version in your terminal
    ```
2. Create a virtual environment named venv using the following command. Note this will create venv folder in the root directory of the project
     ```python3 -m venv venv
    source venv/bin/activate
    ```
3. Activate the virtual environment. The command to do this will depend on your operating system. 
   * On Unix or MacOS, you use:
        ```
        source venv/bin/activate
        ```
    * On window
      ```
        .\venv\Scripts\activate
        ``` 
4. Once the virtual environment is activated
    ```
    pip install -r  requirements.txt
    ``` 
## Pinecone API Key 
    *  Follow the instruction here (https://docs.pinecone.io/docs/quickstart) 
    *  once you have the api key, write it on the .env file for e.g. PINECONE_API_KEY = f4c9c474-73e7-4a4e-9dae-4c96a4bb084a

## Running the app using Flask 
    python main.py

Hit the endpoint on 127.0.0.1:5000

## Running the app from terminal using the arguments
### upload data 
python upload_data.py --index_name "my-index" --urls "http://example1.com" "http://example2.com" 
Note the index_name is the index of the vector database where you will store all the information. Name must consist of lower case alphanumeric characters or '-'.
### query data
python query_data.py --query "why the sky is blue?"

