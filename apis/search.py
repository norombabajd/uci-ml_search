import gensim, spacy

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from flask_restx import Namespace, Resource, fields
import env as env

# UPDATE env.py CSV PATH
data = pd.read_csv(env.CSV_PATH)


data['Abstract'].fillna("", inplace=True)
data['Abstract'] = np.where(data['Abstract'] == "Provide all relevant information about your data set.", "", data['Abstract'])
data['description'] = data['Name']+" "+data['Abstract']
all_titles = [x.lower() for x in list(data['Name'].values)]
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

def _helper(sentence : str, values : set):
        nlp = spacy.load('en_core_web_lg')
        key = nlp(sentence)
        for i, title in enumerate(all_titles):
            test = nlp(title)
            if key.similarity(test) > 0.6:
                values.add((title, key.similarity(test), i))
        for i, info in enumerate(data['Abstract']):
            if type(info) != float:
                test = nlp(info)
                if key.similarity(test) > 0.6:           #threshold
                    values.add((all_titles[i], key.similarity(test), i))

@api.route('/search_semantic')
class Search(Resource):
    @api.expect(payload)
    @api.marshal_with(response, code=200)
    def post(self):
        sentence = self.api.payload['sentence'].lower()

        ds = set()
        _helper(sentence, ds)
        
        def feed():
            container = []
            lda_model = gensim.models.ldamodel.LdaModel.load("C:\\Users\\loren\\uci-ml_search-main\\model\\search.model")
            topics = {i:lda_model.print_topic(i) for i in range(0, 250)}
            for val in topics.values():
                for string in val.split(" + "):
                    container.append(string[7:].rstrip('"'))
            return container 
        
        def val_search(it, key):
            nlp = spacy.load('en_core_web_md')
            key = nlp(key)
            temp = []
            for i in it:
                test = nlp(i)
                if (key.similarity(test) > 0.6):
                    temp.append((i, key.similarity(test)))
            temp = sorted(temp, key=lambda x: x[1], reverse=True)[:5]
            return temp
        res = list(ds)
        res = sorted(res, key=lambda x: x[1], reverse = True)
        l = val_search(feed(),sentence)
        for i in l:
            temp_set = set()
            _helper(i[0], temp_set)
            te = list(temp_set)
            res = res + te

        results = [{"Data Name" : data['Name'].values[x[2]], "Data Description" : data['Abstract'].values[x[2]]} for x in res[:30]]
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
        results = [{"Data Name" : data['Name'].values[x], "Data Description" : data['Abstract'].values[x]} for x in ranking[:10]]
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
                results.append({"Data Name" : data['Name'].values[i], "Data Description" : data['Abstract'].values[i]})
        if results:
            return {'results': results}, 200
        else:
            return {'results': [{"Data Name" : "", "Data Description" : ""}]}, 200