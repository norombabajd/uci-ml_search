import gensim, re, env, pandas as pd
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from nltk.corpus import stopwords

# Update common list of stopwords
#import nltk
#nltk.download('stopwords')

#C:\Users\loren\anaconda3\envs\search\python.exe

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# Format relevant information from datasets
papers = pd.read_csv(env.CSV_PATH).drop(columns=['dataset_id', 'acknowledgement'])
papers['paper_text_processed'] = papers['relevant_info'].map(lambda x: re.sub('[,\.!?]', '', str(x)))
papers['paper_text_processed'] = papers['relevant_info'].map(lambda x: str(x).lower())

# Process sentences into words
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence),deacc=True))

# Remove "stopwords" from data
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
            if word not in stop_words] for doc in texts]

# Format & Assign IDs
data = papers.paper_text_processed.values.tolist()
data_words = remove_stopwords(list(sent_to_words(data)))
id2word = corpora.Dictionary(data_words)

def get_corpus():
    texts = data_words
    return [id2word.doc2bow(text) for text in texts]

def get_ida2word():
    return id2word