import env, gensim, pyLDAvis
import lda_prepare as lda
import pyLDAvis.gensim_models as gensim_models
import sys
import lda_vis as ldav

#cluster visualization
'''
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
import pandas as pd
import matplotlib.colors as mcolors
import numpy as np
'''

#Dominant Topics Visualization



if __name__ == '__main__':

    # Latent Dirichlet Allocation model training using Gensim Multicore (10 topics)
    corpus, id2word, data_words = lda.get_corpus(), lda.get_id2word(), lda.get_data_words()

    #if len(sys.argv) != 3:
    #    raise Exception("Usage: <#topics> <#cores>")
    try:
        syscores = int(sys.argv[2])
        systopics = int(sys.argv[1])
    except:
        syscores = 1
        systopics = 10

    if syscores > 1:
        print("Using " + str(syscores) + " cores, training MULTIcore model with " + str(systopics) + " topics")
        lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=systopics, workers=syscores)
    else:
        print("Training SINGLEcore model with " + str(systopics) + " topics")
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=systopics, update_every=0, passes=20)
    
    print('build SUCCESSFUL, visualizing...')
    # Visualizations using pyLDAvis
    
    LDAvis_prepared = gensim_models.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(LDAvis_prepared, env.VISUALIZATION_PATH)
    
    topic_sents_keywords = ldav.format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_words)
    dominant_topic = topic_sents_keywords.reset_index()
    dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    dominant_topic.head(10)
    print(type(dominant_topic))
    # Cluster Visualization
    '''
    topic_weights = []
    for i, row_list in enumerate(lda_model[corpus]):
        topic_weights.append([w for i, w in row_list[0]])
    
    arr = pd.DataFrame(topic_weights).fillna(0).values
    arr = arr[np.amaz(arr, axis=1) > 0.35]

    topic_num = np.argmax(arr, axis=1)

    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(arr)

    output_notebook()
    n_topics = 4
    mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
    plot = figure(title="t-SNE Clustering of {} LDA Topics".format(systopics),
                  plot_width=900, plot_height=700)
    plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
    show(plot)'''