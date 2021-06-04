#!/usr/bin/env python
# coding: utf-8

# In[11]:


from LDA_simplified import LDA


# In[12]:


lda = LDA(5,'InfovisVAST-papers.jig.txt')


# Build up the nodes and edges for the cytoscape network graph

# In[13]:


# extract the color from class settings: will be defined in stylesheet
# the label is the top 4 words
# position defined by lda
def get_graph_topic_nodes():
    return [{'data': {'id': id_, 'label': vals[0]},
             'classes': f'topic_{vals[1]}'                                                                                        #,"opacity": 0.1
            ,'position':{'x' : vals[2][0] , 'y' : vals[2][1]}
            } for id_,vals in lda.get_topic_nodes_().items()                        
        ]


# In[14]:


# document nodes: id by document id, class defined in stylesheet,
# define the color and belonging
def get_graph_document_nodes():
    return [                       
            {'data': {'id': id_,'size': 1000}, 
             'style': {'shape': 'circle'},
             'classes': vals[1]
             #vals[1]
            } for id_,vals in lda.get_doc_nodes_().items()                        
        ]


# In[15]:


# document edges based on cosine similarity
def get_graph_cos_sim_edges():
        return [
            {'data': {'source': f[0], 'target': f[1],'label': f'{f[0]} -> {f[1]}'}} for f in lda.get_edges_()
        ]


# In[16]:


# edges between the  topics and the related documents: they are invisible
def get_doc_topic_edges():
    return  [
            {'data': {'source': id_, 'target': vals[2],'label': f'{id_} -> {vals[2]}',"edgeLength":200, 'size': 5},
            'style': {'line-color': 'white', "opacity": 0}}  for id_,vals in lda.get_doc_nodes_().items() 
        ]


# In[17]:


# Update stylessheet:  define all the new clusters class settings
# after the update_lda step e.g.
def update_stylesheet():
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
                 
                           
            


# In[18]:


#prepare the data for the cluster summary view
def build_cluster_summary_view():
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


# In[19]:


#Prepare the checklist for the merge cluster functionality
def build_cluster_merge_list():
    clusters = []
    for k in lda.get_topic_nodes.keys():
        clusters.append({'label': k, 'value': k.replace('Cluster ', '')})

    return clusters


# In[20]:


initial_sum_view = build_cluster_summary_view()
from jupyter_dash import JupyterDash  #  pip install jupyter-dash
import dash_cytoscape as cyto  # pip install dash-cytoscape==0.2.0 or higher
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
from dash.dependencies import Output, Input,State



import dash_bootstrap_components as dbc #pip install dash-bootstrap-components

import dash
import plotly.graph_objects as go
import dash_dangerously_set_inner_html
from wordcloud import WordCloud
import base64
from io import BytesIO

def plot_wordcloud(number_of_clusters = 20):
    clust_id =lda.get_last_selected_cluster()
    words = lda.get_top_n_word_probs_for_topic_i(clust_id, number_of_clusters)['Words']
    probs = lda.get_top_n_word_probs_for_topic_i(clust_id, number_of_clusters)['Probabilities']

    data = {a : x for a, x in zip(words, probs)}
    wc = WordCloud(background_color='white', width=480, height=360)
    wc.fit_words(data)
    #lda.get_colormap_for_cluster(clust_id)
    wc.recolor(colormap=lda.get_colormap_for_cluster())
    return wc.to_image()


zero_margin_layout = layout = go.Layout(
  margin=go.layout.Margin(
        l=0, #left margin
        r=0, #right margin
        b=0, #bottom margin
        t=0,  #top margin
        pad=0
    )
)


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# SKETCHY
app = JupyterDash(__name__, external_stylesheets=[dbc.themes.MATERIA], suppress_callback_exceptions=True)


app.layout = html.Div([html.Div(html.H1('iVisClustering: An Interactive Visual Document Clustering via Topic Modeling', style={'backgroundColor':'lightgray'})),
    ############## FIRST ROW ########################
    dbc.Row([
        dbc.Col([
            #html.Div(id='empty-div', children=''),

            #html.P(id='cytoscape-mouseoverNodeData-output'),
            dcc.Location(id='url', refresh=True),
            html.H3('Cluster Relation View'),   
            html.Br(),
            html.A(html.Button('Reset settings', id = 'reset_button', n_clicks = 0),href='/'),
            html.Br(),
            html.I("Number of clusters (min: 2 , max: 10)"),
            dcc.Input(id="input1", type="number", min = 2, value = 5, max = 10),
            dbc.Modal([dbc.ModalHeader("Warning!"),
                       dbc.ModalBody("Number of clusters should be within 2 and 10"),
                       dbc.ModalFooter(dbc.Button(
                                            "Close", id="close_warning_num_of_cluster", className="ml-auto"
                                        )
                                    ),
                       ],
            id="number_of_topics_warning",
            centered=True),
            html.Button("Update", id="update_button", n_clicks = 0),
            html.Br(),
            html.I("Cosine similarity:"),
            html.Div(id="output"),
                dcc.Slider(
                    id='my-daq-slider-ex',
                    min=0, max=1, value=0.4, step = 0.05
                ),
                #html.Br(),
                #dcc.Interval(id='refresh', interval=1),
            html.Div([
                html.I("Delete marked node/document:"),
               html.Button("Delete", id="delete_button",  n_clicks = 0)],
                style={'padding': '30px 10px 10px 10px'}),
            #dcc.Interval(id='refresh', interval=1),
            html.Br(),
            html.Div([
                    html.P(id='cytoscape-tapNodeData-output')],
                    id = 'click_node_div_desc',
                    style={'width': 300,'height' : 150,'padding': '5px 10px 5px 5px'}
                        #,"border":"2px black solid"}
                )
        ],width = 2),
            dbc.Modal([dbc.ModalHeader("Warning!"),
                           dbc.ModalBody("You have to select a document (node in graphs, except document summary) to delete"),
                               dbc.ModalFooter(dbc.Button(
                                                "Close", id="close_delete_document_warning", className="ml-auto"
                                            )
                                        ),
                            ],
                id="delete_document_warning",
                centered=True),

        dbc.Col([
           cyto.Cytoscape(
                id='cytoscape',
                minZoom=0.2,
                maxZoom=2,
                autoRefreshLayout = True,                
                layout={'name': 'cose', 'animate': True},        
                style={'width': '100%', 'height': '400px'},
                elements=  get_graph_topic_nodes() + get_graph_document_nodes() + get_graph_cos_sim_edges() + get_doc_topic_edges(),
                stylesheet = update_stylesheet()
        
        
    )



    ], width = 3),
        ###### intitialise the place for the cluster summary view,
        ###### update it with a callback
        dbc.Col([html.Div([
            html.H3('Cluster Summary View'),
            html.Div([dcc.Graph(id = 'clust_sum_graph',
                                #style={'width': '40vh', 'height': '40vh',},
                                style={'width': '100%', 'height': '40vh',},
                                figure= go.Figure(data = go.Treemap(labels  = initial_sum_view['labels'],
                                                                    parents = initial_sum_view['parents'],
                                                                    marker_colors =  initial_sum_view['marker_colors'],
                                                                    text = initial_sum_view['text_info']),
                                                  layout = go.Layout(margin={'t': 0, 'l': 0, 'r': 5, 'b' : 2})))])
            ,html.Div([ html.Button("Delete Cluster", id="delete_cluster_button",  n_clicks = 0),
                        dbc.Modal([
                                    dbc.ModalHeader("Warning!"),
                                    dbc.ModalBody([
                                        html.Div("You have to select a cluster first from the Cluster summary view"),
                                        html.Div("AND"),
                                        html.Div("Number of clusters should remain at least 2")
                                        ]),
                                    dbc.ModalFooter(
                                        dbc.Button(
                                            "Close", id="close-warning", className="ml-auto"
                                        )
                                    ),
                                ],
                                    id="cluster_delete_failed_warning",
                                    centered=True,
        )])])
        ], width = 2),
        dbc.Col([html.Div([html.I("Merge clusters:"),
                            html.Br(),
                           dcc.Checklist(
                                options = build_cluster_merge_list(),
                                id = 'cluster_merge_checklist'
                           ),
                           html.Button("Merge Clusters", id="merge_cluster_button",  n_clicks = 0),
                           dbc.Modal([
                                        dbc.ModalHeader("Warning!"),
                                        dbc.ModalBody("You have to select one or more (at most k-1) clusters from the checklist, and the number of clusters must remain at least 2"),
                                        dbc.ModalFooter(
                                            dbc.Button(
                                                "Close", id="close_cluster_merge_warning", className="ml-auto"
                                            )
                                        ),
                                    ],
                            id="cluster_merge_failed_warning",
                            centered=True,
            )])],width = 1),

        dbc.Col([html.Div([
            html.Div([],id='bar_chart')
        ], style= {'display': 'none'}, id = 'term_weight_barplot_div')], width = 2)
    ],
    style = {'padding': 10}
    ),

     ## paralell coordinates plot
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H3('Parallel Coordinates View'),
                dcc.Graph(id='parallel_coord',
                          style={'width': '100vh', 'height': '30vh'},
                          figure = go.Figure(data=
                                            go.Parcoords(
                                                line = dict(color = lda.get_parall_coord_df()['Dominant_Topic'],
                                                           colorscale = list(lda.get_colors().values())[:lda.get_k()]),
                                                dimensions = [
                                                    dict(range = [0,1],
                                                        label = f'Cluster {i}',
                                                        values = lda.get_parall_coord_df()[i])
                                                     for i in range(lda.get_k())
                                                ]

                                            ),
              layout = zero_margin_layout
                ))],id = 'parall_coord_div',  style={'padding': '60px 10px 5px 5px'})
        ], width = 5),
        dbc.Col([
             html.Div([html.I("Set threshold for the probability, that a document belongs to the cluster :"),
                        html.I("Filter out noisy documents"),
                        dcc.Slider(
                            id='pc_slider',
                            min=0, max=1, value=0, step = 0.05
                        )],style={'padding': '10px 20px 5px 5px'})
        ], width = 2),
        dbc.Col([html.Div([
            html.H3('Word Cloud'),
            #html.Div([],id='word_cloud')
            #html.Div([],id='word_cloud')
            html.Img(id="word_cloud")
        ], style= {'display': 'none'}, id = 'word_cloud_style')], width = 5)
         
    ]),
    ########### THIRD ROW #####################    
                     
    dbc.Row([
        dbc.Col([
                html.Div([
            html.Div([],id='dt')

        ], style= {'display': 'none'}, id = 'dt_input')], width = 8)
        ])], style={'padding': '5px 20px 20px 20px'})

#])    

  

######### Number of clusters ################    


@app.callback(    
    Output('cytoscape', 'stylesheet'),        
    Output('cytoscape','elements'),
    Output('cytoscape','layout'),
    Output('parallel_coord','figure'),
    Output('clust_sum_graph','figure'),
    Output('cluster_delete_failed_warning','is_open'),
    Output('number_of_topics_warning','is_open'),
    Output('delete_document_warning','is_open'),
    Output('cluster_merge_failed_warning','is_open'),
    Output('cluster_merge_checklist','value'),
    Output('cluster_merge_checklist','options'),    
    Input('my-daq-slider-ex', 'value'),
    Input("update_button", "n_clicks"),
    Input('delete_button', "n_clicks"),
    Input('cytoscape','tapNodeData'),
    Input('cytoscape', 'selectedNodeData'),
    Input('pc_slider','value'),
    Input('clust_sum_graph','clickData'),
    Input('reset_button',"n_clicks"),
    Input('delete_cluster_button', "n_clicks"),
    Input('merge_cluster_button', "n_clicks"),
    Input("close-warning", "n_clicks"),
    Input("close_warning_num_of_cluster", "n_clicks"),
    Input("close_delete_document_warning", "n_clicks"),
    Input("close_cluster_merge_warning", "n_clicks"),
    State('parallel_coord','figure'),
    State("input1", "value"),
    State('cytoscape','stylesheet'),
    State('cytoscape', 'elements'),
    State('cytoscape', 'layout'),
    State('clust_sum_graph','figure'),
    State('clust_sum_graph','clickData'),
    State('cluster_delete_failed_warning','is_open'),
    State('number_of_topics_warning','is_open'),
    State('delete_document_warning','is_open'),
    State('cluster_merge_failed_warning','is_open'),
    State('cluster_merge_checklist','value'),
    State('cluster_merge_checklist','options'),

    prevent_initial_call = True
    
)
def update_graph(value_slider, update_n_button,delete_button, tapNodeData, selectedNodeData,
                 pc_slider,clust_sum_data,reset_button,delete_cluster_button,merge_cluster_button,clust_delete_warn_button,
                 close_warning_num_of_cluster,close_delete_document_warning,close_cluster_merge_warning,
                 pc_figure,cluster_number_value, stylesheet, elements,layout,clust_sum_graph, clust_sum_latest,
                 cluster_delete_failed_warning,number_of_topics_warning,delete_document_warning,cluster_merge_warning,
                 cluster_merge_checklist_vals, cluster_merge_checklist_opts):

    ctx = dash.callback_context
    figure = pc_figure
    clust_sum_figure = clust_sum_graph
    cluster_delete_warn_dialog = cluster_delete_failed_warning
    update_cluster_warn_dialog = number_of_topics_warning
    document_delete_warn_dialog = delete_document_warning
    close_cluster_merge_warning_dialog = cluster_merge_warning
    clicked_element = ctx.triggered[0]['prop_id'].split('.')[0]
    clust_opts = cluster_merge_checklist_opts 

    clust_vals = cluster_merge_checklist_vals if cluster_merge_checklist_vals is not None else []

    #################### cosine similarity ############################
    if  clicked_element == 'my-daq-slider-ex':
        slider_value =  ctx.triggered[0]['value']
        lda.set_cosine_sim_treshold(slider_value)
        ## need to update the elements of the graph
        elements = get_graph_topic_nodes() + get_graph_document_nodes() + get_graph_cos_sim_edges() + get_doc_topic_edges()

    #################  number of clusters ##############################    
    elif clicked_element == 'update_button':
        if cluster_number_value is not None:
            ### elements need to be updated
            lda.set_number_of_clusters(cluster_number_value)
            lda.update_lda()
            elements = get_graph_topic_nodes() + get_graph_document_nodes() + get_graph_cos_sim_edges() + get_doc_topic_edges()
            ### layout of the graph needs to be updated
            layout = {'name': 'cose'}

            ### parallel coordinates plot needs to be updated
            ### according to the number of clusters
            figure=figure = go.Figure(data=
                go.Parcoords(
                    line = dict(color = lda.get_parall_coord_df()['Dominant_Topic'],
                               colorscale = list(lda.get_colors().values())[:lda.get_k()]),
                    dimensions = [
                        dict(range = [0,1],
                            label = f'Cluster {i}',
                            values = lda.get_parall_coord_df()[i])
                         for i in range(lda.get_k())
                    ]),
            layout = zero_margin_layout)

            ### update stylesheet
            stylesheet = update_stylesheet()

            #update cluster summary view
            update_clust_summary = build_cluster_summary_view()
            clust_sum_figure =  go.Figure(go.Treemap(labels  = update_clust_summary['labels'],
                                                         parents = update_clust_summary['parents'],
                                                         marker_colors =  update_clust_summary['marker_colors'],
                                                         text = update_clust_summary['text_info']),
                                              layout = go.Layout(margin={'t': 0, 'l': 0, 'r': 15, 'b' : 2}))

                                                   
            clust_opts = build_cluster_merge_list() 
            clust_vals = []

            lda.set_last_selected_cluster_from_clust_sum_view(None)
            lda.set_last_selected_cluster(None)

        else: #show warning dialog if the input value is out of the limit
            update_cluster_warn_dialog = True
    ################### highlight rows #################################    
    # the choosen document will be highlighted on the parallel coordinates plot
    elif clicked_element == 'cytoscape' and tapNodeData is not None: 
        ## recolor the choosen line
        color_list = list(lda.get_colors().values())[:lda.get_k()]
        color_list.append('black')
        df = lda.get_parall_coord_df()
        df = df.reset_index()
        df['index'] = range(1, len(df) + 1)
        df.set_index('index')
        df.loc[tapNodeData['id'],'Dominant_Topic'] = lda.get_k()

        '''
        figure= go.Figure(data= 
            go.Parcoords(
                line = dict(color = df['Dominant_Topic'],
                           colorscale = color_list),
                dimensions = [                       
                    dict(range = [0,1],
                        label = f'Cluster {i}', 
                        values = df[i])
                     for i in range(lda.get_k())                      
                ]),
        layout = zero_margin_layout)
        '''
    ################## delete documents #################################
    elif clicked_element == 'delete_button':
        #If no node selected before, the input variable should be None, we have to handle it
        if (tapNodeData is not None and 'Cluster' not in tapNodeData['id']):
            lda.remove_document(int(tapNodeData['id']))
            elements = get_graph_topic_nodes() + get_graph_document_nodes() + get_graph_cos_sim_edges() + get_doc_topic_edges()

        else: #Show warning dialog
            document_delete_warn_dialog = True
            

    ######### Filter the paralell coordinates by the given threshold #####
    elif clicked_element == 'pc_slider':
            #print('pc_slider triggered')
            lda.filter_parall_coords_topic_contribution(ctx.triggered[0]['value'])
            #TODO: Filter out documents as well
            #parall_coord_input = lda.get_filtered_topics_df()
            #filtered_parall_coords = lda.get_parall_coord_df().loc[lda.get_filtered_topics_df()['Title']]
            elements = get_graph_topic_nodes() + get_graph_document_nodes() + get_graph_cos_sim_edges() + get_doc_topic_edges()
            filtered_parall_coords = lda.get_filtered_parall_coords_df()
            figure = go.Figure(data= go.Parcoords(
                line = dict(color = filtered_parall_coords['Dominant_Topic'],
                           colorscale = list(lda.get_colors().values())[:lda.get_k()]),
                dimensions = [                       
                    dict(range = [0,1],
                        label = f'Cluster {i}', 
                        values = filtered_parall_coords[i])
                     for i in range(lda.get_k())                      
                ]),
            layout = zero_margin_layout)

            stylesheet = update_stylesheet()

    elif clicked_element == 'clust_sum_graph':
        #print('clust_sum_graph')
        print(clust_sum_data)
        #print(clust_sum_data['points'][0]['label'])
        lda.set_last_selected_cluster_from_clust_sum_view(clust_sum_data['points'][0]['label'])
        #set currently selected cluster from summary view in class aas actual in order to delete
        
    elif clicked_element == 'delete_cluster_button':
        if clust_sum_latest is None  or lda.get_k() == 2 or lda.get_last_selected_cluster_from_clust_sum_view() is None:
            #show the warning that no cluster selected before in the cluster summary view
            cluster_delete_warn_dialog = True
        else:            
            #print(clust_sum_latest['points'][0]['label'])

            lda.delete_cluster()
            elements = get_graph_topic_nodes() + get_graph_document_nodes() + get_graph_cos_sim_edges() + get_doc_topic_edges()
            ### layout of the graph needs to be updated
            layout = {'name': 'cose'}

            ### parallel coordinates plot needs to be updated
            ### according to the number of clusters
            figure=figure = go.Figure(data=
                go.Parcoords(
                    line = dict(color = lda.get_parall_coord_df()['Dominant_Topic'],
                               colorscale = list(lda.get_colors().values())[:lda.get_k()]),
                    dimensions = [
                        dict(range = [0,1],
                            label = f'Cluster {i}',
                            values = lda.get_parall_coord_df()[i])
                         for i in range(lda.get_k())
                    ]),
                layout = zero_margin_layout)

            ### update stylesheet
            stylesheet = update_stylesheet()

            #update cluster summary view
            update_clust_summary = build_cluster_summary_view()
            clust_sum_figure =  go.Figure(go.Treemap(labels  = update_clust_summary['labels'],
                                                     parents = update_clust_summary['parents'],
                                                     marker_colors =  update_clust_summary['marker_colors'],
                                                     text = update_clust_summary['text_info']),
                                          layout = go.Layout(margin={'t': 0, 'l': 0, 'r': 5, 'b' : 2}))
            

            clust_opts = build_cluster_merge_list()
            clust_vals = []

            lda.set_last_selected_cluster_from_clust_sum_view(None)
            lda.set_last_selected_cluster(None)

            #print(clust_opts)
    elif clicked_element == "merge_cluster_button":
            if cluster_merge_checklist_vals is None or not(1 < len(cluster_merge_checklist_vals) < lda.get_k()):
                close_cluster_merge_warning_dialog = True
            else:
                lda.merge_cluster(cluster_merge_checklist_vals)
                elements = get_graph_topic_nodes() + get_graph_document_nodes() + get_graph_cos_sim_edges() + get_doc_topic_edges()
            ### layout of the graph needs to be updated
                layout = {'name': 'cose'}

                ### parallel coordinates plot needs to be updated
                ### according to the number of clusters
                figure = go.Figure(data=
                    go.Parcoords(
                        line = dict(color = lda.get_parall_coord_df()['Dominant_Topic'],
                                   colorscale = list(lda.get_colors().values())[:lda.get_k()]),
                        dimensions = [
                            dict(range = [0,1],
                                label = f'Cluster {i}',
                                values = lda.get_parall_coord_df()[i])
                             for i in range(lda.get_k())
                        ]),
                    layout = zero_margin_layout)

                ### update stylesheet
                stylesheet = update_stylesheet()

                #update cluster summary view
                update_clust_summary = build_cluster_summary_view()
                clust_sum_figure =  go.Figure(go.Treemap(labels  = update_clust_summary['labels'],
                                                         parents = update_clust_summary['parents'],
                                                         marker_colors =  update_clust_summary['marker_colors'],
                                                         text = update_clust_summary['text_info']),
                                              layout = go.Layout(margin={'t': 0, 'l': 0, 'r': 5, 'b' : 2}))


                clust_opts = build_cluster_merge_list()
                clust_vals = []

                lda.set_last_selected_cluster_from_clust_sum_view(None)
                lda.set_last_selected_cluster(None)


    elif clicked_element in ["close-warning","close_warning_num_of_cluster","close_delete_document_warning","close_cluster_merge_warning"]:
            if clicked_element == "close-warning" : cluster_delete_warn_dialog = False
            elif clicked_element == "close_warning_num_of_cluster": update_cluster_warn_dialog = False
            elif clicked_element == "close_cluster_merge_warning": close_cluster_merge_warning_dialog = False
            else: document_delete_warn_dialog = False

    elif clicked_element == 'reset_button':
        lda.reset_settings()
        elements = get_graph_topic_nodes() + get_graph_document_nodes() + get_graph_cos_sim_edges() + get_doc_topic_edges()
        ### layout of the graph needs to be updated
        layout = {'name': 'cose'}

        ### parallel coordinates plot needs to be updated
        ### according to the number of clusters
        figure=figure = go.Figure(data=
            go.Parcoords(
                line = dict(color = lda.get_parall_coord_df()['Dominant_Topic'],
                           colorscale = list(lda.get_colors().values())[:lda.get_k()]),
                dimensions = [
                    dict(range = [0,1],
                        label = f'Cluster {i}',
                        values = lda.get_parall_coord_df()[i])
                     for i in range(lda.get_k())
                ]),
            layout = zero_margin_layout)

        ### update stylesheet
        stylesheet = update_stylesheet()

        #update cluster summary view
        update_clust_summary = build_cluster_summary_view()
        clust_sum_figure =  go.Figure(go.Treemap(labels  = update_clust_summary['labels'],
                                                     parents = update_clust_summary['parents'],
                                                     marker_colors =  update_clust_summary['marker_colors'],
                                                     text = update_clust_summary['text_info']),
                                          layout = go.Layout(margin={'t': 0, 'l': 0, 'r': 15, 'b' : 2}))

        clust_opts = build_cluster_merge_list()
        clust_vals = []

    else:
        return dash.no_update


            
    return [stylesheet, elements,layout, figure,clust_sum_figure,cluster_delete_warn_dialog,update_cluster_warn_dialog,document_delete_warn_dialog,close_cluster_merge_warning_dialog,clust_vals,clust_opts]

@app.callback(
    Output('empty-div', 'children'),
    Input('cytoscape', 'mouseoverNodeData'),
    Input('cytoscape','mouseoverEdgeData'),
    Input('cytoscape','tapEdgeData'),
    Input('cytoscape','tapNodeData'),
    Input('cytoscape','selectedNodeData')
)
def update_layout(mouse_on_node, mouse_on_edge, tap_edge, tap_node, snd):
    print("Mouse on Node: {}".format(mouse_on_node))
    print("Mouse on Edge: {}".format(mouse_on_edge))
    print("Tapped Edge: {}".format(tap_edge))
    print("Tapped Node: {}".format(tap_node))
    print("------------------------------------------------------------")
    print("All selected Nodes: {}".format(snd))
    print("------------------------------------------------------------")

    return 'see print statement for nodes and edges selected.'



############################  Term-weight view   #######################
@app.callback(
    Output('bar_chart', 'children'),
    Output('term_weight_barplot_div', 'style'),
    Output('word_cloud_style', "style"),
    #Output('word_cloud', "children"),
    Output('word_cloud', "src"),
    Input('cytoscape', 'tapNodeData'),
    Input("update_button", "n_clicks"),
	Input('delete_cluster_button', "n_clicks"),
    Input('merge_cluster_button', "n_clicks"),
    State("input1", "value"),
    State('clust_sum_graph','clickData'),
    State('cluster_merge_checklist','value'),


    prevent_initial_call = True
)
def update_barplot(tapNodeData,update_button,delete_cluster_button,merge_cluster_button, cluster_num_value,clust_sum_latest,cluster_merge_checklist_vals):

    ctx = dash.callback_context
    clicked_element = ctx.triggered[0]['prop_id'].split('.')[0]
    style = {'display': 'block'}
    figure = []
    word_cloud= ''

    if clicked_element == "update_button":
        if cluster_num_value is None:
            return dash.no_update                
        style = {'display': 'none'}
        
    elif clicked_element == 'delete_cluster_button':
        if clust_sum_latest is None  or lda.get_k() == 2 or lda.get_last_selected_cluster_from_clust_sum_view() is None:
            
            return dash.no_update
        
        style = {'display': 'none'}
        
    elif clicked_element == 'merge_cluster_button':
        if cluster_merge_checklist_vals is None or not(1 < len(cluster_merge_checklist_vals) < lda.get_k()):
            return dash.no_update
        style = {'display': 'none'}
        
    elif 'Cluster' in tapNodeData['id']:
        cluster_id = int(tapNodeData['id'].replace('Cluster ',''))
        lda.set_last_selected_cluster(cluster_id)
        
        data = lda.get_top_n_word_probs_for_topic_i(cluster_id).sort_values(by = "Probabilities",
                                                                            ascending = False)
        table_data = data.to_dict('records')
        ## change table, according to the cluster choosen

        ## change the barplot, according to the cluster chosen
        bar_color = [lda.get_colors()[lda.get_last_selected_cluster()]] * len(data)
        fg = px.bar(lda.get_top_n_word_probs_for_topic_i(cluster_id).round(decimals=4),
                        x = "Probabilities",
                        y = "Words",
                        text = 'Probabilities',
                        #textposition='auto',
                        color_discrete_sequence = bar_color,
                        orientation='h'
                        )
        fg.update_layout(
            margin=dict(l=0, r=0, t=0, b=0))

        figure=html.Div([dcc.Graph(id='horizontal_bar_plot', 
                      style={'width': '200%', 'height': '450px'},
                      figure = fg)
                    ])

        img = BytesIO()
        plot_wordcloud().save(img, format='PNG')
        word_cloud = 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

    else:
        return dash.no_update



    return [figure,style, style, word_cloud]


    
    
######## Show the documents from a choosen cluster in the document view #########

@app.callback(
    Output('dt', 'children'),
    Output('dt_input', 'style'),
    Input('cytoscape', 'tapNodeData'),
    Input("update_button", "n_clicks"),
    Input('delete_cluster_button', "n_clicks"),
    Input('merge_cluster_button', "n_clicks"),
    Input('delete_button', "n_clicks"),
    State("input1", "value"),
    State('clust_sum_graph','clickData'),
    State('cluster_merge_checklist','value'),
    
    prevent_initial_call = True    
)
def update_result(tapNodeData,update_button,delete_cluster_button,merge_cluster_button,delete_document_button,cluster_num_value,clust_sum_latest,cluster_merge_checklist_vals):
    ctx = dash.callback_context
    clicked_element = ctx.triggered[0]['prop_id'].split('.')[0]
    child = []
    if clicked_element == "update_button":
        if cluster_num_value is None:
            return dash.no_update
        
        style = {'display': 'none'}
        
    elif clicked_element == 'delete_cluster_button':
        if clust_sum_latest is None  or lda.get_k() == 2 or lda.get_last_selected_cluster_from_clust_sum_view() is None:            
            return dash.no_update
        
        style = {'display': 'none'}
        
    elif clicked_element == 'merge_cluster_button':
        if cluster_merge_checklist_vals is None or not(1 < len(cluster_merge_checklist_vals) < lda.get_k()):
            return dash.no_update
        
        style = {'display': 'none'}
    
    elif clicked_element == 'delete_button':
        if (tapNodeData is not None and 'Cluster' not in tapNodeData['id']):
            style = {'display': 'none'}
        
        else:
            return dash.no_update
        
    elif tapNodeData is not None and 'Cluster' not in tapNodeData['id']:
        node_title = lda.get_document_title_by_id(int(tapNodeData['id']))
        data=lda.get_data()[lda.get_data()['title'] == node_title]
        doc_with_higlighted_terms = lda.build_term_higlights(data)
        child = html.Div(children = [dash_dangerously_set_inner_html.DangerouslySetInnerHTML(f'''<h2>{node_title}</h2>'''),
                        html.Div([dash_dangerously_set_inner_html.DangerouslySetInnerHTML(doc_with_higlighted_terms)])])
        style = {'display': 'block'}
        
    else: 
        return dash.no_update
    
    return [child,style] 
    


@app.callback(Output('cytoscape-tapNodeData-output', 'children'),
               Output('click_node_div_desc', 'style'),
                Input('cytoscape', 'tapNodeData'),                
                Input("update_button", "n_clicks"),
                Input('delete_cluster_button', "n_clicks"),
                Input('merge_cluster_button', "n_clicks"),
                State('click_node_div_desc', 'style'),
                State("input1", "value"),
                State('clust_sum_graph','clickData'),
                State('cluster_merge_checklist','value'),

                prevent_initial_call = True
              )
def displayTapNodeData(tapNodeData,update_button,delete_cluster_button,merge_cluster_button, style_current,cluster_num_value,clust_sum_latest,cluster_merge_checklist_vals):
    tap_data = tapNodeData['id']
    style = style_current
    ctx = dash.callback_context
    clicked_element = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if clicked_element == "update_button":
        if cluster_num_value is None:
            return dash.no_update                
        style = {'display': 'none'}
        
    elif clicked_element == 'delete_cluster_button':
        if clust_sum_latest is None  or lda.get_k() == 2 or lda.get_last_selected_cluster_from_clust_sum_view() is None:            
            return dash.no_update
        
        style = {'display': 'none'}
        
    elif clicked_element == 'merge_cluster_button':
        if cluster_merge_checklist_vals is None or not(1 < len(cluster_merge_checklist_vals) < lda.get_k()):
            return dash.no_update
        style = {'display': 'none'}
        
    elif 'Cluster' not in tap_data:
        style["backgroundColor"] = f'{lda.get_color_with_opacity(tap_data, True)}'
        return ["You recently clicked/tapped the document: " + lda.get_document_nodes()[tap_data][0],style]
    else:
        style["backgroundColor"] = f'{lda.get_color_with_opacity(tap_data, False)}'
        return ["You recently clicked/tapped cluster: " + tap_data ,style]
    
    return ['', style]

app.run_server( port=8051, dev_tools_hot_reload=True)


# In[20]:




