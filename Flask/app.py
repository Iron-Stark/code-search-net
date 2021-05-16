import nmslib
import torch
from flask import Flask, render_template, request, url_for
from sentence_transformers import SentenceTransformer
import pandas as pd
import re

import autoencoder

app = Flask(__name__)

valid_dataframe = pd.read_csv('../generated_resources/valid_data.csv')
albert_model = SentenceTransformer('bert-base-nli-mean-tokens').to('cuda')
autoencoder_model = autoencoder.AutoEncoder(768, 256).to('cuda')
autoencoder_model.load_state_dict(torch.load('../generated_resources/autoencoder_0.pt'))
autoencoder_model.eval()
search_index = nmslib.init(method='hnsw', space='cosinesimil')
search_index.loadIndex('../generated_resources/final.nmslib')


def search(query):
    embedding = albert_model.encode(query)
    with torch.no_grad():
        output = autoencoder_model(torch.tensor(embedding).to('cuda'))
    # Search five nearest neighbours, their index value and cosine distances are returned
    idxs, dists = search_index.knnQuery(output.cpu(), k=5)

    # Function details for the index value returned are extracted and printed
    all_funcs = []
    list_of_dist = []
    list_of_git = []
    for idx, dist in zip(idxs, dists):
        code = valid_dataframe['code'][idx]
        list_of_git.append(valid_dataframe['url'][idx])
        list_of_dist.append(dist)
        code = re.sub(r'"""(.*)?"""\s\n', r' ', code, flags=re.DOTALL)
        all_funcs.append(code)
    return all_funcs, list_of_dist, list_of_git


@app.route('/')
def main_page():
    return render_template("main_page.html")


@app.route('/results', methods=['GET'])
def results_page():
    query = request.args.get('query')
    funcs, dists, gits = search(query)
    print(funcs)
    values = len(funcs)
    return render_template("result_page.html", data=query, result=values, codes=funcs, dist=dists, git=gits)


if __name__ == "__main__":
    app.run()
