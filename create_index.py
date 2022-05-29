from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import nmslib

data = pd.read_csv('data/datasets.csv')
data['relevant_info'].fillna("", inplace=True)
data['relevant_info'] = np.where(data['relevant_info'] == "Provide all relevant information about your data set.", "", data['relevant_info'])
data['description'] = data['dataset_title']+" "+data['relevant_info']
sentences = list(data['description'].values)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
index = nmslib.init(method='hnsw', space='cosinesimil')
index.addDataPointBatch(embeddings)
index.createIndex(print_progress=True)
index.saveIndex("data_index")

