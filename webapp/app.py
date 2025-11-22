import dash
import kagglehub
import numpy as np
import pandas as pd
from kagglehub import KaggleDatasetAdapter

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio 
from plotly.subplots import make_subplots
import math

from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State

pio.templates.default = "plotly"

################### DATA PREPARATION ###################
# Download the latest dataset from kaggle
# downloaded_path = kagglehub.dataset_download("thedevastator/spotify-tracks-genre-dataset")

# Load the dataset using dataset_load
#songs = kagglehub.dataset_load(
#  KaggleDatasetAdapter.PANDAS,
#  "thedevastator/spotify-tracks-genre-dataset",
#  "train.csv",
#)

# For local testing, load from local CSV
songs = pd.read_csv('webapp/data/train.csv')

# outline of audio features and genres
audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
genres = songs['track_genre'].unique().tolist()


# Pair audio features and popularity column and genres
pop_v_features = audio_features + ['popularity', 'duration_ms'] + ['track_genre']
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

def filter_by_genres(df, genres_selected):
    """Return df filtered by selected genres (list); if empty/None, return original."""
    if genres_selected:
        return df[df['track_genre'].isin(genres_selected)]
    return df

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
        template='plotly',
        color_discrete_sequence=['#1ed760']
    )
    fig.update_layout(
        bargap=0.25,
        bargroupgap=0.05,
        margin=dict(l=160, r=30, t=60, b=40),
        xaxis_title=f'Average {feature.capitalize()}',
        yaxis_title='Genre',
        showlegend=False,
        height=FEATURE_BAR_HEIGHT,
        autosize=True,
        plot_bgcolor='rgb(215, 195, 223)'
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
    fig.update_layout(legend_title_text='Popularity Bucket', plot_bgcolor='rgb(215, 195, 223)')
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
    fig.update_layout(legend_title_text='Popularity Bucket', plot_bgcolor='rgb(215, 195, 223)')
    return fig

def build_duration_line(df, opacity, bin_seconds=30, min_count=5):
    dff = df.copy()
    dff = dff[dff['duration_ms'].notna() & dff['popularity'].notna()]
    # convert to seconds
    dff['duration_sec'] = dff['duration_ms'] / 1000.0

    # create bins (0 .. max_duration) with width bin_seconds
    max_sec = max(1, int(dff['duration_sec'].max()))
    bins = np.arange(0, max_sec + bin_seconds, bin_seconds)
    # label bins by their midpoint (in seconds)
    bin_labels = [(bins[i] + bins[i+1]) / 2.0 for i in range(len(bins)-1)]
    dff['duration_bin'] = pd.cut(dff['duration_sec'], bins=bins, labels=bin_labels, include_lowest=True)
    agg = (dff.groupby('duration_bin', as_index=False)
           .agg(mean_popularity=('popularity', 'mean'), count=('popularity', 'size')))

    # drop empty bins and low-count bins
    agg = agg[agg['duration_bin'].notna()]
    if agg.empty:
        return go.Figure(layout_title_text="No binned duration data")
    agg['duration_bin'] = agg['duration_bin'].astype(float)
    agg = agg[agg['count'] >= min_count]
    if agg.empty:
        # if filtering by min_count removed everything, relax and show all bins
        agg = (dff.groupby('duration_bin', as_index=False)
               .agg(mean_popularity=('popularity', 'mean'), count=('popularity', 'size')))
        agg = agg[agg['duration_bin'].notna()]
        agg['duration_bin'] = agg['duration_bin'].astype(float)
    # convert seconds to minutes
    agg['duration_min'] = agg['duration_bin'] / 60.0

    fig = px.line(
        agg.sort_values('duration_min'),
        x='duration_min',
        y='mean_popularity',
        markers=True,
        title='Mean Popularity vs Track Duration (minutes)',
        labels={'duration_min': 'Duration (minutes)', 'mean_popularity': 'Mean Popularity'},
        template='plotly'
    )
    fig.update_traces(hovertemplate='Duration: %{x:.2f} min<br>Mean Popularity: %{y:.2f}<br>Count: %{customdata[0]}',
                      customdata=agg[['count']].values)
    fig.update_traces(opacity=opacity)
    fig.update_layout(plot_bgcolor='rgb(215, 195, 223)')
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
    fig.update_layout(boxmode='group', plot_bgcolor='rgb(215, 195, 223)')
    return fig

MAX_FACETS = 25
FACET_BASE_HEIGHT = 1000
FACET_WRAP_FIXED = 5  
def build_facet(df, x_axis):
    dff = df.copy()
    gs = sorted(dff['track_genre'].dropna().unique().tolist())
    capped = False
    if len(gs) > MAX_FACETS:
        gs = gs[:MAX_FACETS]; capped = True
    dff = dff[dff['track_genre'].isin(gs)]

    # Shared bins across all genres
    dff = dff[dff['duration_ms'].notna() & dff['popularity'].notna()]
    max_sec = max(1, int(dff['duration_ms'].max() / 1000.0))
    bin_seconds = 30
    bins = np.arange(0, max_sec + bin_seconds, bin_seconds)
    # Label bins by their midpoint (in seconds)
    bin_labels = [(bins[i] + bins[i+1]) / 2.0 for i in range(len(bins)-1)]

    ncols = max(1, min(FACET_WRAP_FIXED, len(gs)))
    nrows = math.ceil(len(gs) / ncols)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=gs,
                        shared_yaxes=True, horizontal_spacing=0.03, vertical_spacing=0.12)

    for i, genre in enumerate(gs):
        row = (i // ncols) + 1
        col = (i % ncols) + 1
        gdf = dff[dff['track_genre'] == genre].copy()
        if gdf.empty:
            continue
        gdf['duration_sec'] = gdf['duration_ms'] / 1000.0
        gdf['duration_bin'] = pd.cut(gdf['duration_sec'], bins=bins, labels=bin_labels, include_lowest=True)

        agg = (gdf.groupby('duration_bin', as_index=False)
               .agg(mean_popularity=('popularity', 'mean'), count=('popularity', 'size')))
        agg = agg[agg['duration_bin'].notna()]
        if agg.empty:
            continue
        agg['duration_bin'] = agg['duration_bin'].astype(float)
        agg['duration_min'] = agg['duration_bin'] / 60.0

        # Filter bins with very low counts to reduce noise
        min_count = 2
        disp = agg[agg['count'] >= min_count]
        if disp.empty:
            disp = agg

        # Marker sizes based on counts
        max_count = disp['count'].max()
        sizes = disp['count'] / max_count * 15 + 5  # scale to [5,20]

        fig.add_trace(
            go.Scatter(
                x=disp['duration_min'],
                y=disp['mean_popularity'],
                mode='lines+markers',
                marker=dict(size=sizes, color='rgb(30, 214, 90)', opacity=0.4,),
                line=dict(width=1.5, color='rgb(29, 185, 84)'),
                hovertemplate='Duration: %{x:.2f} min<br>Mean Popularity: %{y:.2f}<br>Count: %{customdata[0]}',
                customdata=disp[['count']].values,
                showlegend=False
            ),
            row=row, col=col
        )

        xaxis_name = f'xaxis{(row-1)*ncols + col}' if not (row == 1 and col == 1) else 'xaxis'
        fig.layout[xaxis_name].update(title='Duration (min)')
        yaxis_name = f'yaxis{(row-1)*ncols + col}' if not (row == 1 and col == 1) else 'yaxis'
        fig.layout[yaxis_name].update(title='Mean Popularity')

    fig.update_layout(
        height=max(FACET_BASE_HEIGHT, 300 * nrows),
        title_text=(f'Duration vs Mean Popularity by Genre' + (f' (first {MAX_FACETS})' if capped else '')),
        margin=dict(l=40, r=10, t=80, b=40),
        plot_bgcolor='rgb(215, 195, 223)'
    )

    return fig

def build_table(df, selected_bucket):
    dff = df.copy()
    if selected_bucket:
        dff = dff[dff['popularity_bucket'] == selected_bucket]
    columns = ['track_genre', 'popularity', 'popularity_bucket'] + audio_features
    head_df = dff.head(25)
    table = go.Figure(data=[go.Table(
        header=dict(values=[c.capitalize() for c in columns],
                    fill_color="#000000", font=dict(color='white', size=12),
                    align='left'),
        cells=dict(values=[head_df[c] for c in columns],
                   fill_color=['#1ed760'], align='left'))
    ])
    table.update_layout(title=f'Sample Records (n={len(head_df)})')
    return table

################### DASH APP ##############################
app = Dash(__name__)
app.layout = html.Div([
    html.Div([
        html.H1("Spotify Audio Features & Popularity Analysis", style={'textAlign': 'center'}),
        html.P("Interactive exploration of audio features for Spotify tracks with respect to popularity and genres.", style={'textAlign': 'center'}),
        html.Label("Select Audio Feature:", className='dropdown-labels'),
        dcc.Dropdown(
            id='global-feature',
            options=[{'label': f.capitalize(), 'value': f} for f in audio_features],
            value=audio_features[0],
            clearable=False,
            className='dropdown',
            placeholder="Pick an audio feature"
        ),
        html.Label("Filter Genres (multi):", className='dropdown-labels'),
        dcc.Dropdown(
            id='genre-filter',
            options=[{'label': g, 'value': g} for g in sorted(genres)],
            value=[],
            multi=True,
            className='dropdown',
            placeholder="Select one or more genres"
        ),
        html.Button('Apply Genre Filter', id='apply-genre-filter', n_clicks=0),
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
            dcc.Tab(label="Average Feature Values for Each Genre", children=[
                html.Div([
                    dcc.Graph(id='feature-bar', style={'height': f'{FEATURE_BAR_HEIGHT}px', 'width': '100%'}, config={'responsive': True}),
                ])
            ]),
            dcc.Tab(label="Audio Feature vs Popularity", children=[
                html.Div([
                    html.Label("Adjust Point Opacity:", className='dropdown-labels'),
                    dcc.Slider(
                        id='scatter-opacity',
                        min=0.1, max=1.0, step=0.1, value=0.5,
                        marks={round(x,1): str(round(x,1)) for x in np.linspace(0.1,1.0,10)}
                    ),
                    dcc.Graph(id='scatter-fig')
                ])
            ]),
            dcc.Tab(label="Duration vs Mean Popularity", children=[
                html.Div([
                    html.Label("Adjust Line Opacity:", className='dropdown-labels'),
                    dcc.Slider(
                        id='line-opacity',
                        min=0.1, max=1.0, step=0.1, value=0.6,
                        marks={round(x,1): str(round(x,1)) for x in np.linspace(0.1,1.0,10)}
                    ),
                    dcc.Graph(id='line-fig')
                ])
            ]),
            dcc.Tab(label="Audio Feature Distributions", children=[
                html.Div([
                    html.Label("Select Box Orientation:", className='dropdown-labels'),
                    dcc.RadioItems(
                        id='dist-orientation',
                        options=[{'label': 'Vertical', 'value': 'v'},
                                 {'label': 'Horizontal', 'value': 'h'}],
                        value='v',
                        inline=True
                    ),
                    html.Label("Filter Popularity Buckets:", className='dropdown-labels'),
                    dcc.Dropdown(
                        id='dist-pop',
                        options=[{'label': p, 'value': p} for p in popularity_labels],
                        value=[],
                        multi=True,
                        className='dropdown',
                        placeholder="Filter popularity buckets (optional)"
                    ),
                    dcc.Graph(id='dist-fig')
                ])
            ]),
            dcc.Tab(label="Multi-Facet View", children=[
                html.Div([
                    html.Label("Select Genres for Facet (multi):", className='dropdown-labels'),
                    dcc.Dropdown(
                        id='facet-genre-filter',
                        options=[{'label': g, 'value': g} for g in sorted(genres)],
                        value=[],
                        multi=True,
                        className='dropdown',
                        placeholder="Choose genres to include (blank = all)"
                    ),
                    dcc.Graph(
                        id='facet-fig',
                        style={'height': f'{FACET_BASE_HEIGHT}px', 'width': '100%', 'flex': '1'},
                        config={'responsive': True}
                    )
                ])
            ]),
            dcc.Tab(label="Tabularized Data", children=[
                html.Div([
                    html.Label("Filter Popularity Buckets:", className='dropdown-labels'),
                    dcc.Dropdown(
                        id='table-pop-bucket',
                        options=[{'label': p, 'value': p} for p in popularity_labels],
                        value=None,
                        placeholder="Select popularity bucket",
                        className='dropdown'
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
     Input('scatter-opacity', 'value'),
     Input('apply-genre-filter', 'n_clicks'),
     State('genre-filter', 'value')]
)
def update_scatter(x_axis, opacity, apply_clicks, genres_selected):
    dff = filter_by_genres(songs_pvf, genres_selected)
    if dff.empty:
        return go.Figure(layout_title_text="No data for selected genres")
    return build_scatter(dff, x_axis, opacity)

@app.callback(
    Output('line-fig', 'figure'),
    [Input('line-opacity', 'value'),
     Input('apply-genre-filter', 'n_clicks'),
     State('genre-filter', 'value')]
)
def update_line(opacity, apply_clicks, genres_selected):
    dff = filter_by_genres(songs_pvf, genres_selected)
    return build_duration_line(dff, opacity)

@app.callback(
    Output('dist-fig', 'figure'),
    [Input('global-feature', 'value'),
     Input('dist-pop', 'value'),
     Input('dist-orientation', 'value'),
    Input('apply-genre-filter', 'n_clicks'),
     State('genre-filter', 'value')]
)
def update_distribution(feature, selected_pop, orientation, apply_clicks, genres_selected):
    dff = filter_by_genres(songs_pvf, genres_selected)
    return build_box(dff, feature, selected_pop, orientation)

@app.callback(
    Output('facet-fig', 'figure'),
    [Input('global-feature', 'value'),
     Input('facet-genre-filter', 'value')]
)
def update_facet(x_axis, facet_genres):
    # If user selected specific genres, filter; else use all
    dff = songs_pvf
    if facet_genres:
        dff = dff[dff['track_genre'].isin(facet_genres)]
    if dff.empty:
        return go.Figure(layout=dict(title="No data for selected facet genres", height=FACET_BASE_HEIGHT))
    return build_facet(dff, x_axis)

@app.callback(
    Output('table-fig', 'figure'),
    Input('table-pop-bucket', 'value')
)
def update_table(bucket):
    return build_table(songs_pvf, bucket)

if __name__ == '__main__':
    app.run(debug=True)