from pprint import pprint

import env, gensim, pyLDAvis
import lda_prepare as lda
import pyLDAvis.gensim_models as gensim_models


if __name__ == '__main__':

    # Latent Dirichlet Allocation model training using Gensim Multicore (10 topics)
    corpus, ida2word, num_topics = lda.get_corpus(), lda.get_ida2word(), 10
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=ida2word, num_topics=num_topics)
    # pprint(lda_model.print_topics)
    
    # Visualizations using pyLDAvis
    LDAvis_prepared = gensim_models.prepare(lda_model, corpus, ida2word)
    pyLDAvis.save_html(LDAvis_prepared, env.VISUALIZATION_PATH)
