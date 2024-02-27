from src import SemanticSearch
import argparse
semanticSearch = SemanticSearch()

def query(query):
    return semanticSearch.search_relevant_text(query)


if __name__ == "__query__":
    parser = argparse.ArgumentParser(description='Semantic Search')
    parser.add_argument('--query', type=str, required=True, help='Search query')
    args = parser.parse_args()
    query(args.query)