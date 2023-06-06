
import json
import dash
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc

import pandas as pd
import plotly.express as px
from gensim.models import KeyedVectors


def get_visualization_df(embeddings, user_lookup):
    visuals = pd.DataFrame(embeddings, index=words, columns=['x', 'y', 'z'])
    visuals['size'] = 0.01
    visuals['node_type'] = [i.split(':')[0] if not (i.split(':')[1] == 'True') else 'Age' for i in visuals.index.tolist()]
    visuals = visuals.reset_index()

    account_nodes = visuals[visuals['node_type'] == 'Account']
    account_ids = account_nodes['index'].apply(lambda x: x.split(':')[1]).astype(str)
    account_metadata = pd.DataFrame(account_ids.map(user_lookup).values.tolist(), index=account_ids)
    account_metadata.index = [f'Account:{i}' for i in account_metadata.index]
    account_metadata = account_metadata.reset_index()

    final = visuals.merge(account_metadata, on='index', how='left')
    final['Name'] = final['Name'].fillna('')
    final['Followers'] = final['Followers'].fillna(0)
    final['Estimated reach'] = final['Estimated reach'].fillna(0)

    return final


def get_recommendations(selections_dict, model, user_lookup, display_n=25):
    results = []
    for node_id, value in model.most_similar_cosmul(**selections_dict, topn=100000):

        node_parts = node_id.split(':')
        if len(node_parts) != 2:
            node_type = node_parts[0]
            node_key = ' '.join(node_parts[1:])
        else:
            node_type, node_key = node_parts

        if node_type == 'Account':
            try:
                results.append((node_type, user_lookup[node_key]['Name'], value))
            except:
                print(node_key)
        elif node_key == 'True':
            pass
        else:
            results.append((node_type, node_id, value))

    df = pd.DataFrame(results, columns=['EntityType', 'Entity', 'Score'])

    results_tables = []
    for node_type, group_df in df.groupby('EntityType'):
        results_tables.extend([
            html.H4(node_type),
            dbc.Table.from_dataframe(group_df[:display_n], striped=True, bordered=True, hover=False)
        ])
        
    return html.Div(results_tables)


def get_dropdown_options(words, user_lookup):
    non_accounts = [i for i in words if (not i.startswith('Account') and not i.endswith('True'))]
    accounts = [i for i in words if i.startswith('Account')]

    results = {}
    for account in accounts:
        account_id = account.split(':')[1]
        ui_name = f"Account:{user_lookup[account_id]['Name']}"
        results[account] = ui_name

    for non_account in non_accounts:
        results[non_account] = non_account
    
    return results


# APPLICATION SETUP
# ------------------------------------
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server
app.title = 'Neuro-Semantic Search'


# DATA SETUP
# ------------------------------------
with open('user_lookup.json', 'r') as f:
    user_lookup = json.load(f)

final = pd.read_csv('final_app_input_df.csv')

model = KeyedVectors.load_word2vec_format("a.bin")
words = list(model.wv.vocab)
# g2v_embeds = pd.DataFrame([model.wv[i] for i in words], index=words)

# reducer = umap.UMAP(n_components=3, min_dist=0.25, n_neighbors=100)
# embeddings = reducer.fit_transform(g2v_embeds)

# final = get_visualization_df(embeddings, user_lookup)
# final.to_csv('final_app_input_df.csv', index=None)


# LAYOUT
# ------------------------------------
dropdown_options = get_dropdown_options(words, user_lookup)

accounts_layout = dcc.Graph(id="accounts-graph", figure=px.scatter_3d(
    final[final['node_type'] == 'Account'], x='x', y='y', z='z', 
    hover_data=['index', 'node_type', 'Name', 'Followers', 'Estimated reach'],
    size='Estimated reach', size_max=100, 
    opacity=0.5, width=1200, height=1200
    )
)

context_layout = dcc.Graph(id="context-graph", figure=px.scatter_3d(
    final[final['node_type'] != 'Account'], x='x', y='y', z='z', 
    animation_frame='node_type', hover_data=['index', 'node_type'],
    size='size', opacity=0.5, width=1200, height=1200
    )
)

positive_dropdown = dcc.Dropdown(
    options=dropdown_options,
    value= ['City:Lisbon'],
    multi=True,
    id='positive-terms'
)

negative_dropdown = dcc.Dropdown(
    options=dropdown_options,
    value=[],
    multi=True,
    id='negative-terms'
)

search_layout = html.Div([
    html.Div([
        html.P('Positive Entities'),
        positive_dropdown,
        ]),
    html.Div([
        html.P('Negative Entities'),
        negative_dropdown,
    ]),
    html.Br(),

    dcc.Loading(
        id="loading-search-results",
        type="cube",
        children=html.Div([html.Div(id='search-results')])
    )
])

app.layout = html.Div(
    children=[
        dbc.NavbarSimple(
            brand="Search Demo",
            color="primary",
            dark=True,
        ),
        dbc.Card(
            [
                dbc.CardHeader(
                    dbc.Tabs(
                        [
                            dbc.Tab(label="Search", tab_id="tab-1"),
                            dbc.Tab(label="Context", tab_id="tab-2"),
                            dbc.Tab(label="Accounts", tab_id="tab-3")
                        ],
                        id="card-tabs"
                    )
                ),
                dbc.CardBody(html.P(id="card-content", className="card-text")),
            ]
        )
    ],
)

@callback(
    Output("card-content", "children"), 
    Input("card-tabs", "active_tab")
)
def tab_content(active_tab):
    if active_tab == 'tab-3':
        return accounts_layout
    elif active_tab == 'tab-2':
        return context_layout
    elif active_tab == 'tab-1':
        return search_layout
    else:
        return accounts_layout


@callback(
    Output("search-results", "children"), 
    Input("positive-terms", "value"),
    Input("negative-terms", "value")
)
def live_recommendations(positive_list, negative_list):
    selections_dict = {}
    if (not positive_list and not negative_list):
        return html.P('Add entities to receive suggestions...')

    if positive_list:
        selections_dict['positive'] = positive_list

    if negative_list:
        selections_dict['negative'] = negative_list

    return get_recommendations(selections_dict, model, user_lookup)


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8786, use_reloader=True)
