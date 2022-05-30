import gensim, re, env, pandas as pd
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from gensim.models import CoherenceModel
import spacy, pathlib, random

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
if __name__ == '__main__':
    # Format relevant information from datasets
    papers = pd.read_csv(env.CSV_PATH).drop(columns=['dataset_id', 'acknowledgement'])
    papers['description'] = papers['dataset_title']+" "+papers['relevant_info']
    papers['paper_text_processed'] = papers['description'].map(lambda x: re.sub('[,\.!?]', '', str(x)))
    papers['paper_text_processed'] = papers['description'].map(lambda x: str(x).lower())

    # Process sentences into words
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence),deacc=True))


    # Format & Assign IDs
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

    data_words_nostops = remove_stopwords(data_words)
    data_words_bigrams = make_bigrams(data_words_nostops)
    nlp = spacy.load("en_core_web_sm")
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    id2word = corpora.Dictionary(data_lemmatized)
    texts = data_lemmatized
    corpus = [id2word.doc2bow(text) for text in texts]

    def prepare_default():
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                iterations=100,
                                                num_topics=10, 
                                                random_state=100,
                                                update_every=1,
                                                chunksize=2000,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
        # Compute Perplexity
        print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, corpus=corpus, dictionary=id2word, coherence='u_mass')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)

    def prepare_to_file(iterations=100, topics=10, state=100, chunks=2000, passes=10, maxCycles=30):         
        cycle = 10
        container = []
        while cycle <= maxCycles:
            lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                        id2word=id2word,
                                                        iterations=iterations,
                                                        num_topics=cycle, 
                                                        random_state=state,
                                                        update_every=1,
                                                        chunksize=chunks,
                                                        passes=passes,
                                                        alpha='auto',
                                                        per_word_topics=True)

            coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, corpus=corpus, dictionary=id2word, coherence='u_mass')
            perplexity, coherence = lda_model.log_perplexity(corpus), coherence_model_lda.get_coherence()
            data = f"Cycle: {cycle}. Perplexity: {perplexity}, Coherence: {coherence}"
            print(data)
            container.append(data)
            cycle=cycle+1

        with open('collections.txt', 'a+') as file:
            for i in container:
                file.write(i + "\n")

    prepare_to_file()
            
            
                                          