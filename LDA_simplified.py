import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
import string
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from matplotlib import colors as mcolors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd  # pip install pandas


# noinspection SpellCheckingInspection

class LDA:
    """
    Init of class: load and clean the data, and make the first lda clustering with the the given k cluster
    :param: k: desired number of clusters used in LDA , should be at least 2, and 10 at maximum
    :param: path: Load path of the file
    """
    def __init__(self, k, path):

        self.orig_clust_num = k
        self.num_of_clusters = k
        self.data_path = path
        # Some default value
        self.cosine_sim_threshold = 0.3

        self.word_prob_range = 0

        #Build up colors dictionary
        self.cols = {}
        cmaps = ['Blues', 'Oranges', 'Greens', 'Reds', 'Purples', 'copper', 'RdPu', 'Greys', 'YlGn', 'BuGn']
        self.colormaps = {}
        for i, key in enumerate(mcolors.TABLEAU_COLORS.keys()):
            cl = key.replace('tab:', '')
            self.cols[i] = cl
            self.colormaps[cl] = cmaps[i]

        self.colors = self.cols
        self.col = list(self.cols.values())[:self.num_of_clusters]

        self.data_read = self.read_data()
        self.data_filtered = self.filter_data
        self.cleaned_data, self.node_dict = self.clean_lemmatize_data()
        self.lda_dict, self.lda_bag_of_words = self.build_bag_of_words_model()
        self.lda_model = self.get_lda()
        self.lda_most_rel_topics = self.get_most_relevant_topics()

        self.parall_coords = self.get_parall_coord_df()

        self.topic_df = self.format_topics_sentence()

        self.topic_nodes = self.get_topic_nodes
        self.filtered_topic_df = None

        self.document_nodes = self.get_document_nodes()

        self.cos_sim = self.calculate_cosine_similarity()

        self.edges = self.get_filtered_edges()
        
        self.filtered_paarcord_topics_df = None

        self.last_selected_cluster = None
        self.last_selected_cluster_from_clust_sum_view = None
        self.top_topic_for_terms = None
        self.clusters_to_merge = None

        self.get_top_topic_for_words()
        
        self.word_probs = self.get_word_probabilities()

        self.indexed_topic_node_df = self.set_indexed_topic_node_df()

    def read_data(self):
        """
        Read the original dataset with bssoup xml extractor
        :return: data stored in BeautiflSoup instance
        """

        with open(self.data_path, encoding="utf8",
                  errors="ignore") as fl:
            fle = fl.read()
            bs_data = BeautifulSoup(fle, "xml")

        return bs_data

    @property
    def filter_data(self):
        """
         Filter for the paper-defined time interval(1994-2010)
         Separate the title and the document description

         Store in dictionary format:
         k: Title , v: document text

        :return: Filtered data stored in dictionary<title, document text>
        """
        dataset = self.data_read.find_all(True)
            
        filtered_docs = {}
        for tag in dataset:
            try:
                # Filter the years
                date = int(tag.find('year').text)
                if 1994 < date < 2010:
                    doc_text = tag.find('docText').text
                    doc_splitted = doc_text.split('\n')
                    # Fitler if multiple linebreaks separate the title and the text
                    doc_splitted = [d for d in doc_splitted if len(d) > 0]
                    # Extract the title
                    title = doc_splitted[0]
                    # Assign the text  to the title in the dictionary
                    filtered_docs[title] = doc_splitted[1]
            except:
                pass

        return filtered_docs

    def get_k(self):
        """
        :return: number of clusters specified
        """
        return self.num_of_clusters

    def clean_lemmatize_data(self):
        """
        Cleaning of data based on the paper description and with further useful approaches:
            - tokenize
            - removal of numeric characters
            - removal of punctuations
            - removal of stopwords
            - lemmatize the 'cleaned data'

        :return: dict<document_number, title> , dict<title, cleaned_text>
        """

        sw = stopwords.words('english')
        lemmatizer = WordNetLemmatizer()
        punctuations = set(string.punctuation)
        cleaned_texts = {}
        node_dict = {}
        for i, (key, b_u) in enumerate(self.data_filtered.items()):
            #Tokenize the data on the delimiterss
            b_tok = wordpunct_tokenize(
                b_u.replace('Case study', '').replace('VRMosaic', '').replace('Research report', ''))
            #Remove punctuations
            res = [s.translate(str.maketrans('', '', string.punctuation)) for s in b_tok]

            #lowercase all the words
            b_tok = [b.lower() for b in res]

            #Remove empty elements, stopwords, punctuations, and digits
            b_cleaned = [b for b in b_tok if
                         not b in sw and not b in punctuations and not b.isdigit() and b != '' and len(b) > 1]

            #Lemmatize the cleaned words
            b_lemm = [lemmatizer.lemmatize(b) for b in b_cleaned]

            # dictionary - key: title , value: cleaned document
            cleaned_texts[key] = b_lemm

            #node dictionary for the networkd graph: assign id to titles , this will help to extract the node selection
            # to refer to the proper document
            node_dict[i] = key

        return cleaned_texts, node_dict

    def build_bag_of_words_model(self):
        """
        Build the bag of words model from the cleaned data
        And the dictionary of the unique words

        :return: dictonary of all words, bag of words model of the documents
        """

        lda_dictionary = Dictionary(self.cleaned_data.values())
        lda_bag_of_words = [lda_dictionary.doc2bow(c, allow_update=True) for c in self.cleaned_data.values()]

        return lda_dictionary, lda_bag_of_words

    def get_lda(self):
        """
        Model LDA with the given cluster number and the built up bag of words model

        :return: lda model
        """

        lda_model: LdaMulticore = LdaMulticore(corpus=self.lda_bag_of_words,
                                               id2word=self.lda_dict,
                                               num_topics=self.num_of_clusters)

        return lda_model

    def get_most_relevant_topics(self):
        """
        extract the most relevant top 4 terms for the topics (relevant for topic node representation)

        :return:dict<cluster_id, [top_4_terms_for_cluster]>
        """
        term_topics = {}
        for i in range(self.num_of_clusters):
            topic_id = i
            term_topics[topic_id] = []
            for ld in self.lda_model.get_topic_terms(i, topn=4):
                term_topics[topic_id].append(self.lda_dict[ld[0]])

        return term_topics

    def get_col(self):
        """
        :return: cluster colors (extracted from matplotlib colors)
        """
        return self.col

    def get_word_probabilities(self):
        """
        extract all the word probabilities from the lda model for each cluster

        :return: dict<topic_id, [word_probabilitities]>
        """

        word_probs = {}
        for i in range(self.num_of_clusters):
            topic_id = i
            word_probs[topic_id] = []
            for l in self.lda_model.get_topic_terms(i, topn=len(self.lda_dict)):
                word_probs[topic_id].append((self.lda_dict[l[0]], l[1]))

        return word_probs

    def get_parall_coord_df(self):
        """
        Build pandas dataframe for the paralell coordinates view

        :return: pandas dataframe for documents with their dominant topic
        """
        parall_coords = {}
        for i, b in zip(self.cleaned_data.keys(), self.lda_bag_of_words):
            parall_coords[i] = self.lda_model[b]

        parall_coord_df = pd.DataFrame.from_dict({k: dict(v) for k, v in parall_coords.items()})
        parall_coord_df = parall_coord_df.replace(np.nan, 0)
        parall_coord_df = parall_coord_df.transpose()
        parall_coord_df = parall_coord_df.reindex(sorted(parall_coord_df.columns), axis=1)
        parall_coord_df['Dominant_Topic'] = parall_coord_df.idxmax(axis="columns")

        return parall_coord_df

    def color_assign_to_topic(self, x):
        """
        :param x: topic id
        :return: color assigned to topic x (dict)
        """
        return self.colors[x]


    def color_assign_to_topic_with_opacity(self, x):
        """
        Assign color to topics with opacity

        :param x: topic id
        :return: opaque color for term highlighting
        """
        color_with_opacity = list(mcolors.to_rgba(self.colors[x]))
        color_with_opacity[3] = 0.3
        rgba = f'rgba({color_with_opacity[0] * 255}, {color_with_opacity[1] * 255}, {color_with_opacity[2] * 255}, {color_with_opacity[3]})'
        return rgba

    def format_topics_sentence(self):
        """
        build up a pandas dataframe with several useful informations: document - Topic belongings, contribution,
        assigned color keywords

        :return: pd.DataFrame('Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text', 'Title','color')
        """
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row_list in enumerate(self.lda_model[self.lda_bag_of_words]):
            row = row_list[0] if self.lda_model.per_word_topics else row_list
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = self.lda_model.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(
                        pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        titles = self.cleaned_data.keys()
        texts = self.cleaned_data.values()
        document_nums = pd.Series(sorted(self.node_dict.keys()))
        titles = pd.Series(titles)
        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([document_nums, sent_topics_df, contents, titles], axis=1)
        sent_topics_df = sent_topics_df.reset_index(drop = True)
        sent_topics_df.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text', 'Title']
        # Convert dominaant topic to int
        sent_topics_df['Dominant_Topic'] = sent_topics_df['Dominant_Topic'].astype(int)
        sent_topics_df['color'] = sent_topics_df.apply(lambda x: self.color_assign_to_topic(x['Dominant_Topic']),
                                                       axis=1)

        return sent_topics_df

    @property
    def get_topic_nodes(self):
        """
        build dictionary for the topics which will be input for the cytoscape node, generate a random position as well

        :return: <topic_id, (color, position)>
        """
        topic_dict = {}
        for k, v in self.lda_most_rel_topics.items():
            #random position
            pos = np.random.randint(900, size=2)
            topic_id = f'Cluster {k}'
            # key: topic id, value: [top terms for the topic with linebreak, color assigned to topic, position of the
            # topic node]
            topic_dict[topic_id] = (' '.join(v).replace(' ', '\n'), self.colors[k], pos)

        return topic_dict

    def set_indexed_topic_node_df(self):
        """
        topic dataframe in indexed format: for faster searching

        :return: indexed df
        """
        df = self.topic_df.copy()
        df.set_index('Document_No', inplace= True)

        return df

    def get_indexed_topic_node_df(self):
        return self.indexed_topic_node_df

    def get_color_with_opacity(self, id, is_node_id):
        """
        :param id: id of the node /cluster
        :param is_node_id: bool, whether the selected cyoscape element is document node or cluster node
        :return: opaque color for the background
        """
        if is_node_id:
            color = self.color_assign_to_topic_with_opacity(self.get_indexed_topic_node_df().loc[int(id)]['Dominant_Topic'])
        else:
            color = self.color_assign_to_topic_with_opacity(int(str(id).replace('Cluster ','')))

        return color

    def get_colormap_for_cluster(self):
        """
        Build colormap for wordcloud

        :return: colormap related to current cluster (assigned by the already related cluster color)
        """
        return self.colormaps[self.col[int(self.get_last_selected_cluster())]]

    def get_document_nodes(self):
        """
        build dictionary for document nodes to the cytoscape network visualization

        :return: dict<document_id,(document_title, document_color: color of the cluster,cluster)>
        """
        doc_nodes = {}
        if self.filtered_topic_df is None:
            iterable_df = self.topic_df.copy()
        else:
            iterable_df = self.filtered_topic_df.copy()

        for idx, d in iterable_df.iterrows():
            cluster_id = 'Cluster ' + str(d['Dominant_Topic'])
            doc_nodes[str(d['Document_No'])] = (d['Title'], d['color'], cluster_id)

        return doc_nodes

    #
    def calculate_cosine_similarity(self):
        """
        Calculate Cosine similarity between the documents with sklearn implementation
        Also removing the duplicities (taking only the upper trinagle elements from the result)
        In order to avoid duplicate edges on the graph

        :return: cosine_sim_matrix = np.array([number_of_documents,number_of_documents])
        """

        data = []
        #prepare input for the sklearn cosine similarity function
        for k in sorted(self.node_dict.keys()):
            data.append(" ".join(self.cleaned_data[self.node_dict[k]]))

        vec = TfidfVectorizer()
        x = vec.fit_transform(
            data)

        # Calculate the pairwise cosine similarities (depending on the amount of data that you are going to have this
        # could take a while)
        matrix_similarity = cosine_similarity(x)
        # Remove duplicates + diagonal: cosine similarity returns a symmetric matrix, where the diagonal and the
        # lower or upper  triangular is irrelevant
        tril_ind = np.tril_indices(matrix_similarity.shape[0])
        mat_sim_upper = matrix_similarity.copy()
        mat_sim_upper[tril_ind] = -1

        return mat_sim_upper

    def get_filtered_edges(self):
        """
        Get the visible edges (edges between document nodes over the cosine sim threshold

        :return: [(node_0, node_1, cosine_similarity value),...]
        """
        #filter edges based on the given threshold
        filtered_edges = np.argwhere(self.cos_sim >= self.cosine_sim_threshold)

        # get the document ids -> since document removal is also a function, simply returning the index at the matching
        # points are not enough, since then the graph might attempt to create edge with a non-existing document,
        # which throws an error
        doc_ids = sorted(self.node_dict.keys())
        cos_sim_edges = []

        if self.filtered_topic_df is None:
            available_docs = set(self.topic_df['Document_No'].tolist())
        else:
            available_docs = set(self.filtered_topic_df['Document_No'].tolist())

        for f in filtered_edges:
            idx_0 = doc_ids[f[0]]
            idx_1 = doc_ids[f[1]]
            if idx_0 in available_docs and idx_1 in available_docs:
                cos_sim_edges.append((idx_0, idx_1, self.cos_sim[f[0], f[1]]))

        return cos_sim_edges

    def get_doc_nodes_(self):
        return self.get_document_nodes()

    def get_topic_nodes_(self):
        return self.get_topic_nodes

    def get_edges_(self):
        return self.get_filtered_edges()

    def get_parall_coords_(self):
        return self.parall_coords

    def get_colors(self):
        return self.colors

    def get_topics_df(self):
        return self.topic_df
    
    def get_document_title_by_id(self, value):        
        return self.node_dict[value]

    def get_data(self):
        return pd.DataFrame(list(self.data_filtered.items()), columns=['title', 'document'])

    def get_number_of_clusters(self):
        return self.num_of_clusters

    #setup the desired number of clusters
    def set_number_of_clusters(self, value):
        self.num_of_clusters = value

    #Setup new cosine similarity filter based on the slider value
    def set_cosine_sim_treshold(self, value):
        self.cosine_sim_threshold = value
        self.update_cosine_sim()

    def set_word_prob_range(self, value):
        self.word_prob_range = value

    # To remove documents:
    def remove_document(self, value):
        """
        Remove documents by clicking on the delete document button

        :param value: id of the document node
        :return: updated dictionary of nodes, by removing the marked document
        """
        # Update the document list: LDA has to be recalculated

        del self.cleaned_data[self.node_dict[value]]
        del self.node_dict[value]

        self.update_lda()

    def reset_settings(self):
        """
        Reset the the view and the lda class itself with the original cluster number
        """
        self.__init__(self.orig_clust_num, self.data_path)

    def update_lda(self):
        """
        Re-cluster with the new cluster number (reaction of "update" button on the app)
        """
        self.lda_dict, self.lda_bag_of_words = self.build_bag_of_words_model()

        self.lda_model = self.get_lda()

        self.update_lda_related_class_elements()

    def update_cosine_sim(self):
        """
        Update cosine similarites
        """
        self.cos_sim = self.calculate_cosine_similarity()
        self.edges = self.get_filtered_edges()

    def lda_get_lda_model(self):
        return self.lda_model

    def get_lda_model_state(self):
        return self.lda_model.state

    #word to id from the lda id2word, in order to extract the word lambdas by index
    def get_term_invert_dict(self):
        term_invert_dict = {v: k for k, v in self.lda_model.id2word.items()}

        return term_invert_dict

    def get_topic_term_prob_matrix(self):
        topics_terms_proba_matrix = np.apply_along_axis(lambda x: x / x.sum(), 1, self.get_lda_model_state().get_lambda())

        return topics_terms_proba_matrix

    #not used in the final version
    def update_term_topic_weight(self, topic_idx, term, new_term_weight):
        term_idx = self.get_term_invert_dict()[term]
        self.updated_term_topic_proba = self.get_topic_term_prob_matrix().copy()
        self.updated_term_topic_proba[topic_idx, term_idx] = new_term_weight
        return self.updated_term_topic_proba

    # not used in the final version
    #Build up a new state with the updated weights, then make an inference step (not working currently)
    def term_prob_update_lda(self):
        self.lda_model.init_dir_prior(self.updated_term_topic_proba, 'eta')

        self.lda_model.inference(self.lda_bag_of_words)
        self.lda_model.update(self.lda_bag_of_words)

    #
    #
    def filter_parall_coords_topic_contribution(self, value):
        """
        filter paralell coordinates based on the input value (>value has to be kept)
        filtering also the document-topic df to filter in cytoscape

        :param value: paralell coordinate filter threshold
        :return:filtered paralell coordinates and topic dataframe
        """
        self.filtered_paarcord_topics_df = self.get_topics_df().copy()

        #get_indexed_topic_node_df
        self.filtered_paarcord_topics_df = self.filtered_paarcord_topics_df[self.filtered_paarcord_topics_df['Topic_Perc_Contrib'] >= value]
        remained_docs = self.filtered_paarcord_topics_df['Document_No'].tolist()
        self.filtered_topic_df = self.topic_df[self.topic_df['Document_No'].isin(remained_docs)]

    def get_filtered_parall_coords_df(self):        
        return self.get_parall_coord_df().loc[self.filtered_paarcord_topics_df['Title']]

    def get_top_n_word_probs_for_topic_i(self, topic_id, n = 10):
        """
        Term weight table input: extract the top n words for the currently selected topic

        :param topic_id: id of the cluster
        :param n: number of words to be extracted
        :return: dataframe with the words and the related probabilities
        """
        get_word_probs = self.word_probs[topic_id]

        word = []
        probs = []
        
        for g in get_word_probs:
            word.append(g[0])
            probs.append(g[1])
        
        word_probs_df = pd.DataFrame(data={'Words': word, 'Probabilities': probs})
        #get only the top n words
        word_probs_df = word_probs_df.sort_values(by='Probabilities', ascending=False).head(n)
        word_probs_df.sort_values(by='Probabilities', ascending=True, inplace=True)
        
        return word_probs_df

    def set_last_selected_cluster(self, value):
        self.last_selected_cluster = value
    
    def get_last_selected_cluster(self):
        return self.last_selected_cluster

    def get_top_topic_for_words(self):
        """
        build Topic - word - probability df with the related opaque color for term highlight in document view

        :return: pandas.DataFrame('Word', 'Color')
        """
        topic = []
        word = []
        prob = []
        for k, v in self.get_word_probabilities().items():
            for e in v:
                topic.append(k)
                word.append(e[0])
                prob.append(e[1])

        all_word_probs = pd.DataFrame(columns=['Topic', 'Word', 'Probability'])
        all_word_probs['Topic'] = topic
        all_word_probs['Word'] = word
        all_word_probs['Probability'] = prob

        # Sort by probability in descending order
        all_word_probs.sort_values(by='Probability', ascending=False, inplace=True)

        # Drop duplicate terms, keep always the first --> Get only the top topics for term
        all_word_probs_distinct = all_word_probs.drop_duplicates(subset='Word', keep='first')
        all_word_probs_distinct['Color'] = all_word_probs_distinct\
                                            .apply(lambda x: self.color_assign_to_topic_with_opacity(x['Topic']), axis=1)

        all_word_probs_distinct.reset_index(drop=True,inplace=True)
        self.top_topic_for_terms = all_word_probs_distinct.drop(columns=['Topic', 'Probability'])

    def get_terms_with_best_topic(self):
        return self.top_topic_for_terms

    def build_term_higlights(self, doc_input):
        """
        Setup the document view by highlighting the most relevant terms with their topic correspondence

        :param doc_input: Id of the document
        :return: str(): document as string, with the proper html tags assigned for specific terms (will
        be used in the visualization by dash.DangerouslySetInnerHTML())
        """
        #lowercase and lemmatize
        doc_desc = doc_input['document'].tolist()[0]

        b_tok = wordpunct_tokenize(doc_desc)
        # lowercase all the words
        b_low = [b.lower() for b in b_tok]

        lemmatizer = WordNetLemmatizer()
        # Lemmatize the cleaned words
        b_lemm = [lemmatizer.lemmatize(b) for b in b_low]

        doc_highlighted = b_tok

        top_topic_list = self.top_topic_for_terms['Word'].tolist()

        # Assign the the highlighting to the raw text (unlemmatized etc.)
        for i, b in enumerate(b_lemm):
            if b in top_topic_list:
                color = self.top_topic_for_terms[self.top_topic_for_terms['Word'] == b]['Color'].tolist()[0]
                doc_highlighted[i] = doc_highlighted[i].replace(doc_highlighted[i],f"<span style=\"background-color:"
                                                                                   f" {color};\>{doc_highlighted[i]}"
                                                                                   f"</span> ")
        
        doc_string = ' '.join(doc_highlighted)
        return doc_string

    def set_last_selected_cluster_from_clust_sum_view(self, cluster_id):
        self.last_selected_cluster_from_clust_sum_view = cluster_id

    def get_last_selected_cluster_from_clust_sum_view(self):
        return self.last_selected_cluster_from_clust_sum_view

    def set_clusters_to_merge(self, clusters):
        self.clusters_to_merge = clusters

    def get_clusters_to_merge(self):
        return self.clusters_to_merge

    def update_lda_related_class_elements(self):
        """
        update the relevant class components after cluster merge / delete , re-clustering steps
        """
        self.lda_most_rel_topics = self.get_most_relevant_topics()

        self.topic_df = self.format_topics_sentence()

        # update topic positions function call must be here
        self.topic_nodes = self.get_topic_nodes

        self.document_nodes = self.get_document_nodes()

        self.cos_sim = self.calculate_cosine_similarity()

        self.edges = self.get_filtered_edges()

        self.get_top_topic_for_words()

        self.indexed_topic_node_df = self.set_indexed_topic_node_df()

        self.word_probs = self.get_word_probabilities()

        self.set_last_selected_cluster_from_clust_sum_view(None)
        self.set_last_selected_cluster(None)

        self.col = list(self.cols.values())[:self.num_of_clusters]

        self.filtered_topic_df = None

    def delete_cluster(self):
        """
        Remove cluster from lda: modifying the state of the model

        :return: lda with the removed cluster
        """

        cluster_id = int(self.get_last_selected_cluster_from_clust_sum_view().replace('Cluster ', ''))

        new_state = self.get_lda_model_state()

        #get the current word probabilites to topics
        word_probs_stats = new_state.__dict__['sstats']

        #remove the cluster
        word_probs_stats = np.delete(word_probs_stats, cluster_id, 0)

        #Update the state - with the removal of the cluster
        new_state.__dict__['sstats'] = word_probs_stats.astype(np.double)
        #set the reduced cluster number
        self.num_of_clusters = int(new_state.__dict__['sstats'].shape[0])

        # init the lda with cluster number and term dict
        self.lda_model = LdaMulticore(num_topics=self.num_of_clusters, id2word=self.lda_dict)
        # Set the state for the saved one
        self.lda_model.state = new_state

        self.lda_model.inference(self.lda_bag_of_words)

        self.update_lda_related_class_elements()

    def merge_cluster(self, cluster_ids):
        """
        Merge clusters selected from the checklist: sum up the probs at row wise

        :param cluster_ids: cluster ids selected in checklist
        :return: model with merged clusters
        """
        current_state = self.get_lda_model_state()
        new_state  = current_state
        current_lda_stats = current_state.__dict__['sstats']

        clust_id_int = [int(c) for c in cluster_ids]

        merge_clust_array = np.zeros([len(cluster_ids), len(self.lda_dict.keys())])

        for i,c in enumerate(clust_id_int):
            merge_clust_array[i, :] = current_lda_stats[c, :]

        curr_stats_cleaned = np.delete(current_lda_stats,clust_id_int,0)

        #sum up the statistics
        merged_clusters = np.sum(merge_clust_array,axis=0,keepdims=True)

        final_cluster_stats = np.concatenate((curr_stats_cleaned, merged_clusters), axis=0)

        new_state.__dict__['sstats'] = final_cluster_stats

        self.num_of_clusters = int(new_state.__dict__['sstats'].shape[0])

        # init the lda with cluster number and term dict
        self.lda_model = LdaMulticore(num_topics=self.num_of_clusters, id2word=self.lda_dict)
        # Set the state for the saved one
        self.lda_model.state = new_state

        self.lda_model.inference(self.lda_bag_of_words)

        self.update_lda_related_class_elements()