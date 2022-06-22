from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import nmslib
import env

data = pd.read_csv(env.CSV_PATH)

data['Abstract'].fillna("", inplace=True)
data['Abstract'] = np.where(data['Abstract'] == "Provide all relevant information about your data set.", "", data['Abstract'])
data['description'] = data['Name']+" "+data['Abstract']
sentences = list(data['description'].values)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
index = nmslib.init(method='hnsw', space='cosinesimil')
index.addDataPointBatch(embeddings)
index.createIndex(print_progress=True)
index.saveIndex("dataset_index")

