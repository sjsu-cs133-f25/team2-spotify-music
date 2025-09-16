# Spotify Music Dataset Data Card
Dataset Source Link: https://www.kaggle.com/datasets/thedevastator/spotify-tracks-genre-dataset

## Shape
(114000 rows, 21 columns)

## Column Dictionary
- __Unnamed: 0__: int64, acts as integer id for each row  
- __track_id__: string format of object datatype, a track id given by spotify, not all rows are unique  
- __artists__: string format of object datatype, name of the artist  
- __album_name__: string format of object datatype, name of album  
- __track_name__: string format of object datatype, name of track  
- __popularity__: int64, 0-100 value scale of popularity  
- __duration_ms__: int64, duration of the song in milliseconds  
- __explicit__: bool, tells if the song is clean or explicit  
- __danceability__: float64, 0-1 value scale of suitability for dancing  
- __energy__: float64, 0-1 value for intensity of a track  
- __key__: int64, 0-11 value representing musical key of track  
- __loudness__: float64, loudness of track in decibels  
- __mode__: int64, 0(minor) or 1(major) value representing tonal mode of track  
- __speechiness__: float64, 0-1 value representing the presence of spoken word in a track  
- __acousticness__: float64, 0-1 value representing the extent to which a track possesses an acoustic quality  
- __instrumentalness__: float64, 0-1 value representing the likelihood that a track is instrumental  
- __liveness__: float64, 0-1 value representing the presence of an audience during the recording or performance of track  
- __valence__: float64, 0-1 value representing the musical positiveness conveyed by a track  
- __tempo__: float64, tempo of the track in beats per minute  
- __time_signature__: int64, number of beats within each bar of the track  
- __track_genre__: string format of object datatype, represents genre of track (114 unique genres)

## Missing Snapshot
Columns (artists, album_name, track_name) each have a single missing value, less than 0.000001%  
Row 65900 is the row that has missing values for these columns

## Known Quirks
Column (Unnamed: 0) is largely unneeded as the default dataframe index already serves as the index for the table. It is redundant data that can be dropped from the table.
