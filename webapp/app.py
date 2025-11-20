import dash
import kagglehub
import seaborn as sns
import numpy as np
import pandas as pd
import math
from kagglehub import KaggleDatasetAdapter

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio 

from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import pkgutil
import importlib.util

pio.templates.default = "plotly"

################### DATA PREPARATION ###################
# Download the latest dataset from kaggle
downloaded_path = kagglehub.dataset_download("thedevastator/spotify-tracks-genre-dataset")

# Load the dataset using dataset_load
songs = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "thedevastator/spotify-tracks-genre-dataset",
  "train.csv",
)
songs.dropna()

# outline of audio features and genres
audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
genres = songs['track_genre'].unique().tolist()


# Pair audio features and popularity column and genres
pop_v_features = audio_features + ['popularity'] + ['track_genre']
songs_pvf = songs[pop_v_features]

# handle missing values w/ a simple dropna statement
songs_pvf = songs_pvf.dropna()

# Define popularity bucket boundaries and labels
popularity_bins = [0, 20, 40, 60, 80, 100]
popularity_labels = ['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High']

# Create the popularity_bucket column
songs_pvf['popularity_bucket'] = pd.cut(songs_pvf['popularity'], bins=popularity_bins, labels=popularity_labels, include_lowest=True)

# Ensure string labels (avoid pandas Categorical edge-cases in plotly)
songs_pvf['popularity_bucket'] = songs_pvf['popularity_bucket'].astype(str)

# Create table summarizing means of each audio feature for each genre
genre_summary = songs_pvf.groupby('track_genre')[audio_features].mean().reset_index()

#######################  Graphs #########################
FEATURE_BAR_HEIGHT = 2000

def build_feature_bar(data, feature):
    df = data.copy()
    melted = pd.melt(
        df,
        id_vars=['track_genre'],
        value_vars=audio_features,
        var_name='Audio Feature',
        value_name='Average Value'
    )
    filtered = melted[melted['Audio Feature'] == feature]

    # Sort genres by Average Value (descending)
    filtered = filtered.sort_values('Average Value', ascending=False)

    fig = px.bar(
        filtered,
        x='Average Value',
        y='track_genre',
        orientation='h',
        title=f'Average {feature.capitalize()} per Genre',
        color='Audio Feature',
        template='plotly'
    )
    fig.update_layout(
        bargap=0.25,
        bargroupgap=0.05,
        margin=dict(l=160, r=30, t=60, b=40),
        xaxis_title=f'Average {feature.capitalize()}',
        yaxis_title='Genre',
        showlegend=False,
        height=FEATURE_BAR_HEIGHT,
        autosize=True
    )
    fig.update_yaxes(categoryorder='array', categoryarray=filtered['track_genre'].tolist(),
                     tickfont=dict(size=11), automargin=False)
    fig.update_xaxes(automargin=True)
    return fig

def build_scatter(df, x_axis, opacity):
    dff = df.copy()
    fig = px.scatter(
        dff,
        x=x_axis,
        y='popularity',
        color='popularity_bucket',
        opacity=opacity,
        title=f'{x_axis.capitalize()} vs Popularity',
        labels={x_axis: x_axis.capitalize(), 'popularity': 'Popularity'},
        template="plotly"
    )
    return fig

def build_line(df, x_axis, opacity):
    dff = df.copy()
    agg = (dff.groupby([x_axis, 'popularity_bucket'], as_index=False)['popularity']
             .mean()
             .rename(columns={'popularity': 'mean_popularity'})
             .sort_values(by=[x_axis, 'popularity_bucket']))
    if agg.empty:
        return go.Figure(layout_title_text="No data after filters")
    
    agg['popularity_bucket'] = agg['popularity_bucket'].astype(str)
    fig = px.line(
        agg,
        x=x_axis,
        y='mean_popularity',
        color='popularity_bucket',
        markers=True,
        title=f'Mean Popularity vs {x_axis.capitalize()}',
        labels={x_axis: x_axis.capitalize(), 'mean_popularity': 'Mean Popularity', 'popularity_bucket': 'Popularity Bucket'},
        template="plotly"  
    )
    fig.update_traces(opacity=opacity)
    fig.update_layout(legend_title_text='Popularity Bucket')
    return fig

def build_box(df, feature, selected_pop, orientation):
    dff = df.copy()
    if selected_pop:
        dff = dff[dff['popularity_bucket'].isin(selected_pop)]
    # Ensure order of buckets and valid values
    dff = dff[dff[feature].notna() & dff['popularity_bucket'].notna()]
    dff['popularity_bucket'] = dff['popularity_bucket'].astype(str)

    if orientation == 'v':
        x_col, y_col = 'popularity_bucket', feature
    else:
        x_col, y_col = feature, 'popularity_bucket'

    fig = px.box(
        dff,
        x=x_col,
        y=y_col,
        color='popularity_bucket',
        category_orders={'popularity_bucket': popularity_labels},
        points='outliers',
        title=f'Distribution of {feature.capitalize()} by Popularity Bucket',
        labels={
            'popularity_bucket': 'Popularity bucket',
            feature: feature.capitalize()
        },
        template="plotly"
    )
    fig.update_layout(boxmode='group')
    return fig

MAX_FACETS = 25
FACET_BASE_HEIGHT = 1000
FACET_WRAP_FIXED = 6   
def build_facet(df, x_axis):
    dff = df.copy()
    gs = sorted(dff['track_genre'].dropna().unique().tolist())
    capped = False
    if len(gs) > MAX_FACETS:
        gs = gs[:MAX_FACETS]; capped = True
    dff = dff[dff['track_genre'].isin(gs)]
    agg = (dff.groupby(['track_genre', x_axis, 'popularity_bucket'], as_index=False)['popularity']
             .mean()
             .rename(columns={'popularity': 'mean_popularity'}))
    if agg.empty:
        return go.Figure(layout=dict(title="No data after filters", height=FACET_BASE_HEIGHT, autosize=False))
    agg['popularity_bucket'] = agg['popularity_bucket'].astype(str)
    wrap_eff = max(1, min(FACET_WRAP_FIXED, len(gs)))

    fig = px.line(
        agg,
        x=x_axis,
        y='mean_popularity',
        color='popularity_bucket',
        facet_col='track_genre',
        facet_col_wrap=wrap_eff,
        category_orders={'track_genre': gs},
        markers=True,
        title=f'{x_axis.capitalize()} vs Mean Popularity by Genre' + (f' (first {MAX_FACETS})' if capped else ''),
        labels={x_axis: x_axis.capitalize(), 'mean_popularity': 'Mean Popularity', 'track_genre': 'Genre'},
        template="plotly",
        height=FACET_BASE_HEIGHT
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
    fig.update_layout(margin=dict(l=40, r=10, t=60, b=40), autosize=True, uirevision="facet-static")
    return fig

def build_table(df, selected_bucket):
    dff = df.copy()
    if selected_bucket:
        dff = dff[dff['popularity_bucket'] == selected_bucket]
    columns = ['track_genre', 'popularity', 'popularity_bucket'] + audio_features
    head_df = dff.head(25)
    table = go.Figure(data=[go.Table(
        header=dict(values=[c.capitalize() for c in columns],
                    fill_color='#23385c', font=dict(color='white', size=12),
                    align='left'),
        cells=dict(values=[head_df[c] for c in columns],
                   fill_color=['#FFF2CC'], align='left'))
    ])
    table.update_layout(title=f'Sample Records (n={len(head_df)})')
    return table

################### DASH APP ##############################
app = Dash(__name__)
app.layout = html.Div([
    html.Div([
        html.H1("Spotify Audio Features & Popularity Analysis", style={'textAlign': 'center'}),
        html.P("Interactive exploration of audio features, popularity buckets, and genres.", style={'textAlign': 'center'}),
        html.Label("Select Audio Feature:", className='dropdown-labels'),
        dcc.Dropdown(
            id='global-feature',
            options=[{'label': f.capitalize(), 'value': f} for f in audio_features],
            value=audio_features[0],
            clearable=False,
            placeholder="Pick an audio feature"
        ),
        html.Img(src='/assets/spotify_charts.png', id='spotify-img'),
        html.H3("Feature Descriptions"),
        html.P([html.B("Danceability:"), " Describes how suitable a track is for dancing. Higher values suggest more energy and rhythm. (0.0 to 1.0)"]),
        html.P([html.B("Energy:"), " A measure of intensity and activity. High values suggest energetic tracks that are fast, loud, and noisy. (0.0 to 1.0)"]),
        html.P([html.B("Loudness:"), " The overall loudness of a track in decibels (dB). Positive values represent louder songs while negative values suggest quieter ones."]),
        html.P([html.B("Speechiness:"), " Represents the presence of spoken words in a track. Higher values means more words. (0.0 to 1.0)"]),
        html.P([html.B("Acousticness:"), " A confidence measure of whether the track is acoustic. Higher values indicate more acoustic sounds. (0.0 to 1.0)"]),
        html.P([html.B("Instrumentalness:"), " A score that represents the likelyhood that a song is an instrumental. (0.0 to 1.0)"]),
        html.P([html.B("Liveness:"), " A score that detects the presence of an audience in the recording. (0.0 to 1.0)"]),
        html.P([html.B("Valence:"), " A measure of musical positiveness conveyed by a track. Higher values indicate more positive emotions. (0.0 to 1.0)"]),
        html.P([html.B("Tempo:"), " The speed or pace of a given piece, measured in beats per minute (BPM)."]),
    ], id='left-container'),
    html.Div([
        dcc.Tabs([
            dcc.Tab(label="Overview + Details", children=[
                html.Div([
                    dcc.Graph(id='feature-bar', style={'height': f'{FEATURE_BAR_HEIGHT}px', 'width': '100%'}, config={'responsive': True}),
                ])
            ]),
            dcc.Tab(label="Scatter Plot", children=[
                html.Div([
                    dcc.Slider(
                        id='scatter-opacity',
                        min=0.1, max=1.0, step=0.1, value=0.5,
                        marks={round(x,1): str(round(x,1)) for x in np.linspace(0.1,1.0,10)}
                    ),
                    dcc.Graph(id='scatter-fig')
                ])
            ]),
            dcc.Tab(label="Line Chart", children=[
                html.Div([
                    dcc.Slider(
                        id='line-opacity',
                        min=0.1, max=1.0, step=0.1, value=0.6,
                        marks={round(x,1): str(round(x,1)) for x in np.linspace(0.1,1.0,10)}
                    ),
                    dcc.Graph(id='line-fig')
                ])
            ]),
            dcc.Tab(label="Distribution View", children=[
                html.Div([
                    dcc.RadioItems(
                        id='dist-orientation',
                        options=[{'label': 'Vertical', 'value': 'v'},
                                 {'label': 'Horizontal', 'value': 'h'}],
                        value='v',
                        inline=True
                    ),
                    dcc.Dropdown(
                        id='dist-pop',
                        options=[{'label': p, 'value': p} for p in popularity_labels],
                        value=[],
                        multi=True,
                        placeholder="Filter popularity buckets (optional)"
                    ),
                    dcc.Graph(id='dist-fig')
                ])
            ]),
            dcc.Tab(label="Multi-Facet View", children=[
                html.Div([
                    dcc.Graph(
                        id='facet-fig',
                        style={'height': f'{FACET_BASE_HEIGHT}px', 'width': '100%', 'flex': '1'},
                        config={'responsive': True}
                    )
                ])
            ]),
            dcc.Tab(label="Tabularized Data", children=[
                html.Div([
                    dcc.Dropdown(
                        id='table-pop-bucket',
                        options=[{'label': p, 'value': p} for p in popularity_labels],
                        value=None,
                        placeholder="Select popularity bucket"
                    ),
                    dcc.Graph(id='table-fig')
                ])
            ])
        ])
    ], id='right-container')
], id='container')

################### Callback ##############################
@app.callback(
    Output('feature-bar', 'figure'),
    Input('global-feature', 'value')
)
def update_feature_bar(feature):
    return build_feature_bar(genre_summary, feature)

@app.callback(
    Output('scatter-fig', 'figure'),
    [Input('global-feature', 'value'),
     Input('scatter-opacity', 'value')]
)
def update_scatter(x_axis, opacity):
    return build_scatter(songs_pvf, x_axis, opacity)

@app.callback(
    Output('line-fig', 'figure'),
    [Input('global-feature', 'value'),
     Input('line-opacity', 'value')]
)
def update_line(x_axis, opacity):
    return build_line(songs_pvf, x_axis, opacity)

@app.callback(
    Output('dist-fig', 'figure'),
    [Input('global-feature', 'value'),
     Input('dist-pop', 'value'),
     Input('dist-orientation', 'value')]
)
def update_distribution(feature, selected_pop, orientation):
    return build_box(songs_pvf, feature, selected_pop, orientation)

@app.callback(
    Output('facet-fig', 'figure'),
    Input('global-feature', 'value')
)
def update_facet(x_axis):
    return build_facet(songs_pvf, x_axis)

@app.callback(
    Output('table-fig', 'figure'),
    Input('table-pop-bucket', 'value')
)
def update_table(bucket):
    return build_table(songs_pvf, bucket)

if __name__ == '__main__':
    app.run(debug=True)