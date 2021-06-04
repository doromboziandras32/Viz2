#!/usr/bin/env python
# coding: utf-8

# In[12]:


# In[14]:


def get_graph_topic_nodes():
    """
    extract the color from class settings: will be defined in stylesheet
    the label is the top 4 words

    :return: topic nodes in cytoscape format
    """
    return [{'data': {'id': id_, 'label': vals[0]},
             'classes': f'topic_{vals[1]}'                                                                                        #,"opacity": 0.1
            ,'position':{'x' : vals[2][0] , 'y' : vals[2][1]}
            } for id_,vals in lda.get_topic_nodes_().items()                        
        ]


# In[15]:


# document nodes: id by document id, class defined in stylesheet,
# define the color and belonging
def get_graph_document_nodes():
    """
    :return: graph document nodes with the proper coloring in cytoscape format
    """
    return [                       
            {'data': {'id': id_,'size': 1000}, 
             'style': {'shape': 'circle'},
             'classes': vals[1]
             #vals[1]
            } for id_,vals in lda.get_doc_nodes_().items()                        
        ]


# In[16]:



def get_graph_cos_sim_edges():
    """
    :return: edges within document based on cosine similarity
    """
    return [
        {'data': {'source': f[0], 'target': f[1],'label': f'{f[0]} -> {f[1]}'}} for f in lda.get_edges_()
    ]


# In[17]:


# edges between the  topics and the related documents: they are invisible
def get_doc_topic_edges():
    """
    :return: Invisible edges within  document nodes and their dominant topic
    """
    return  [
            {'data': {'source': id_, 'target': vals[2],'label': f'{id_} -> {vals[2]}',"edgeLength":200, 'size': 5},
            'style': {'line-color': 'white', "opacity": 0}}  for id_,vals in lda.get_doc_nodes_().items() 
        ]


# In[18]:


# Update stylessheet:  define all the new clusters class settings
# after the update_lda step e.g.
def update_stylesheet():
    """
    Update stylessheet:  define all the new clusters class settings (colors etc.)

    :return: Updated graph stylesheet
    """
    colors = sorted(set([vals[1] for id_,vals in lda.get_doc_nodes_().items()]))
    #get the new colors for the topics and the nodes
    node_classes = [{
                    'selector': f'.{c}',
                    'style': {
                        'background-color': c
                    }
                } for c in colors]
    
    topic_classes = [{
                    'selector': f'.topic_{c}',
                    'style': { 'border-color': c,
               'border-width': 2,        
               'background-color': 'white',
                'shape': 'rectangle','content': 'data(label)','text-halign':'center',
                'text-valign':'center','text-wrap': 'wrap','width':'label','height':'label'}
                }for c in colors     ]

    return node_classes + topic_classes   # Class selectors


# In[19]:


def build_cluster_summary_view():
    """
    prepare the data for the cluster summary view

    :return: data in cluster summary view feedable format
    """
    clust = 'Clusters'
    labels = [clust]
    parents = ['']
    marker_colors = ["white"]
    text_info = ['']
    for id_,vals in lda.get_topic_nodes_().items():
        labels.append(id_)
        text_info.append(vals[0].replace('\n','<br>'))
        parents.append(clust)
        marker_colors.append(vals[1])
    clust_sum_view = dict()
    clust_sum_view['labels'] =  labels
    clust_sum_view['parents'] =  parents
    clust_sum_view['marker_colors'] =  marker_colors
    clust_sum_view['text_info'] =  text_info
    return  clust_sum_view


# In[20]:


def build_cluster_merge_list():
    """
    Prepare the checklist for the merge cluster functionality

    :return: dash checklist content with proper label
    """
    clusters = []
    for k in lda.get_topic_nodes().keys():
        clusters.append({'label': k, 'value': k.replace('Cluster ', '')})

    return clusters


# In[21]:




def plot_wordcloud(number_of_words = 20):
    '''
    Wordcloud plot

    :param number_of_words: number of words to be plotted (default: 20)
    :return: Wordcloud plot (image, not interactive, but the words are not overlapped)
    '''
    clust_id =lda.get_last_selected_cluster()
    words = lda.get_top_n_word_probs_for_topic_i(clust_id, number_of_words)['Words']
    probs = lda.get_top_n_word_probs_for_topic_i(clust_id, number_of_words)['Probabilities']

    data = {a : x for a, x in zip(words, probs)}
    wc = WordCloud(background_color='white', width=480, height=360)
    wc.fit_words(data)
    #lda.get_colormap_for_cluster(clust_id)
    wc.recolor(colormap=lda.get_colormap_for_cluster())
    return wc.to_image()
