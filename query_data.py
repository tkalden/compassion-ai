from src import PredictionModel
import argparse
predictModel = PredictionModel()

def main(query):
    result = predictModel.search_relevant_text(query)
    prompt = predictModel.generate_prompt_openai(query,result)
    return prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Semantic Search')
    parser.add_argument('--query', type=str, required=True, help='Search query')
    args = parser.parse_args()
    main(args.query)