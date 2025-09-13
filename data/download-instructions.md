Link to dataset download:
https://www.kaggle.com/datasets/thedevastator/spotify-tracks-genre-dataset

Download the dataset to your notebook with the following code:
```
import kagglehub
import seaborn as sns
import matplotlib.pyplot as plt
from kagglehub import KaggleDatasetAdapter

# Download the latest dataset from kaggle
downloaded_path = kagglehub.dataset_download("thedevastator/spotify-tracks-genre-dataset")

# Load the dataset using dataset_load
songs = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "thedevastator/spotify-tracks-genre-dataset",
  "train.csv",
)
```
