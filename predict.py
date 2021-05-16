import nmslib
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

import autoencoder


def search(query):
    embedding = albert_model.encode(query)
    with torch.no_grad():
        output = autoencoder_model(torch.tensor(embedding).to('cuda'))
    # Search five nearest neighbours, their index value and cosine distances are returned
    idxs, dists = search_index.knnQuery(output.cpu(), k=5)

    # Function details for the index value returned are extracted and printed
    for idx, dist in zip(idxs, dists):
        code = train_dataframe['code'][idx]
        url = train_dataframe['url'][idx]
        print(f'cosine dist:{dist:.4f} \n {url}  \n {code} \n---------------------------------\n')


train_dataframe = pd.read_csv('generated_resources/train_data.csv')
albert_model = SentenceTransformer('bert-base-nli-mean-tokens').to('cuda')
autoencoder_model = autoencoder.AutoEncoder(768, 256).to('cuda')
autoencoder_model.load_state_dict(torch.load('generated_resources/autoencoder_0.pt'))
autoencoder_model.eval()
search_index = nmslib.init(method='hnsw', space='cosinesimil')
search_index.loadIndex('generated_resources/final.nmslib')
search('trains a k nearest neighbors classifier for face recognition')
