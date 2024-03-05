from src import TrainModel
import argparse
trainModel = TrainModel()


def main(index_name, urls):
    return trainModel.upload_data_to_pinecone([], urls, index_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Semantic Search')
    parser.add_argument('--index_name', type=str, required=True, help='Index name')
    parser.add_argument('--urls', type=str, nargs='+', required=True, help='List of URLs')
    args = parser.parse_args()
    main(args.index_name, args.urls)
