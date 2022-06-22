# Scrapes, Formats and Trains a Latent Dirichlet Allocation (LDA) model to be paired with LSA keyword search results.


import gensim, re, env as env, spacy, warnings, pandas as pd
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from gensim.models import CoherenceModel

warnings.filterwarnings("ignore", category=DeprecationWarning)

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

if __name__ == '__main__':

    # Map relevant information from datasets, cut punctuation, and set all letters to lowercase

    # UPDATE env.py FILE WITH ACCURATE CSV PATH BEFORE RUNNING    
    papers = pd.read_csv('donated_datasets.csv').drop(columns=["ID","userID","introPaperID","Types","DOI","DateDonated","isTabular","URLFolder","URLReadme","URLLink","Graphics","Status","NumHits","NumInstances","NumAttributes","slug"])
    # keep: Name, Abstract, Area, Task, AttributeTypes
    papers['description'] = papers['Name']+" "+papers['Abstract']+" "+papers['Area']+" "+papers['Task']+" "+papers['AttributeTypes']
    papers['paper_text_processed'] = papers['description'].map(lambda x: re.sub('[,\.!?]', '', str(x)))
    papers['paper_text_processed'] = papers['description'].map(lambda x: str(x).lower())

    # Process sentences into words
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence),deacc=True))


    data = papers.paper_text_processed.values.tolist()
    data_words = list(sent_to_words(data))

    #Make bigram/trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words])
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) 
                if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # Format, process and lemmatize data
    data_words_nostops = remove_stopwords(data_words)
    data_words_bigrams = make_bigrams(data_words_nostops)
    nlp = spacy.load("en_core_web_sm")
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Construct dictionary & corpus
    id2word = corpora.Dictionary(data_lemmatized)
    texts = data_lemmatized
    corpus = [id2word.doc2bow(text) for text in texts]

    # Build LDA Model (single-core)
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            iterations=100,
                                            num_topics=13, 
                                            random_state=100,
                                            update_every=1,
                                            chunksize=2000,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)

    # Save Model
    lda_model.save('model\\LDAsearch.model')

    # Optional Model Evaluation Steps

    # Compute Model Perplexity
    #perplexity = lda_model.log_perplexity(corpus)  # a measure of how good the model is. lower the better.

    # Measure Coherence Score
    #coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, corpus=corpus, dictionary=id2word, coherence='u_mass')
    #coherence_lda = coherence_model_lda.get_coherence()

    
            
                                          