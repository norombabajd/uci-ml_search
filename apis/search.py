import nmslib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from flask_restx import Namespace, Resource, fields


data = pd.read_csv('C:\\Users\\johndaniel\\search\\data\\datasets.csv')
data['relevant_info'].fillna("", inplace=True)
data['relevant_info'] = np.where(data['relevant_info'] == "Provide all relevant information about your data set.", "", data['relevant_info'])
data['description'] = data['dataset_title']+" "+data['relevant_info']
all_titles = [x.lower() for x in list(data['dataset_title'].values)]
sentences = [x.lower() for x in list(data['description'].values)]
tokenized_corpus = [doc.split(" ") for doc in sentences]
bm25 = BM25Okapi(tokenized_corpus)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
api = Namespace('SearchDataset', description='Search Dataset')


payload = api.model('Search', {
    'sentence': fields.String(required=True, description='Search query')
})

response_fields = api.model('Data', {
    'Data Name': fields.String(required=True, description="Name of the dataset"),
    'Data Description': fields.String(required=True, description="Description of the dataset")
})


response = api.model('Search Response', {
    'results': fields.List(fields.Nested(response_fields))
}
)

@api.route('/search_semantic')
class Search(Resource):
    @api.expect(payload)
    @api.marshal_with(response, code=200)
    def post(self):
        sentence = self.api.payload['sentence'].lower()
        index = nmslib.init(method='hnsw', space='cosinesimil')
        index.loadIndex("C:\\Users\\johndaniel\\search\\data\\data_index")
        embed = model.encode(sentence)
        ids, distances = index.knnQuery(embed, k=10)
        results = [{"Data Name" : data['dataset_title'].values[x], "Data Description" : data['relevant_info'].values[x]} for x in ids]
        return {'results': results}, 200

@api.route('/search_bm25')
class Search(Resource):
    @api.expect(payload)
    @api.marshal_with(response, code=200)
    def post(self):
        sentence = self.api.payload['sentence'].lower()
        tokenized_query = sentence.split(" ")
        doc_scores = bm25.get_scores(tokenized_query)
        ranking = np.argsort(doc_scores)
        results = [{"Data Name" : data['dataset_title'].values[x], "Data Description" : data['relevant_info'].values[x]} for x in ranking[:10]]
        return {'results': results}, 200

@api.route('/search_exact')
class Search(Resource):
    @api.expect(payload)
    @api.marshal_with(response, code=200)
    def post(self):
        sentence = self.api.payload['sentence'].lower()
        results = []
        for i, title in enumerate(all_titles):
            if sentence in title:
                results.append({"Data Name" : data['dataset_title'].values[i], "Data Description" : data['relevant_info'].values[i]})
        if results:
            return {'results': results}, 200
        else:
            return {'results': [{"Data Name" : "", "Data Description" : ""}]}, 200