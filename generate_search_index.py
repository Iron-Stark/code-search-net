import csv

import nmslib
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import time


def generate_function_vectors(data_type):
    func_vector = []
    model = SentenceTransformer('bert-base-nli-mean-tokens').to('cuda')
    model.eval()
    dataframe = pd.read_csv(f'generated_resources/{data_type}_data.csv')
    start = time.time()
    for code in dataframe['code_filtered']:
        try:
            func_vector.append(model.encode(code))  # Store the list of function vectors
        except:
            func_vector.append(np.random.randn(768))
            print(code)
        if len(func_vector) % 1000 == 0:
            end = time.time()
            print(f"Time taken for {len(func_vector)} elements is {end - start}")

    with open(f"generated_resources/{data_type}_func_vectors.tsv", "w+", newline='') as my_csv:
        csvwriter = csv.writer(my_csv, delimiter='\t')
        csvwriter.writerows(func_vector)


def generate_search_index(data_type):
    generate_function_vectors(data_type)
    search_index = nmslib.init(method='hnsw', space='cosinesimil')
    e = np.loadtxt(f"generated_resources/{data_type}_func_vectors.tsv",
                   delimiter='\t')  # Load our saved fucntion vectors
    search_index.addDataPointBatch(e)
    search_index.createIndex(print_progress=True)
    search_index.saveIndex('generated_resources/final.nmslib')  # Save the search indices


if __name__ == '__main__':
    generate_search_index('train')