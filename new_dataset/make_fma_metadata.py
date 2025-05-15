import os
import pandas as pd
import urllib.request
import zipfile
import numpy as np

SPECTROGRAM_DIR = 'fma_spectrograms'
METADATA_CSV = 'fma_metadata.csv'
TRACKS_CSV = 'tracks.csv'
FMA_METADATA_URL = 'https://os.unil.cloud.switch.ch/fma/fma_metadata.zip'
FMA_METADATA_ZIP = 'fma_metadata.zip'

# Download and extract tracks.csv if not present
if not os.path.exists(TRACKS_CSV):
    print('tracks.csv not found. Downloading FMA metadata...')
    if not os.path.exists(FMA_METADATA_ZIP):
        print('Downloading fma_metadata.zip...')
        urllib.request.urlretrieve(FMA_METADATA_URL, FMA_METADATA_ZIP)
        print('Downloaded fma_metadata.zip')
    with zipfile.ZipFile(FMA_METADATA_ZIP, 'r') as zip_ref:
        print('Extracting fma_metadata/tracks.csv from fma_metadata.zip...')
        zip_ref.extract('fma_metadata/tracks.csv', '.')
        # Move and rename to tracks.csv in current directory
        os.rename('fma_metadata/tracks.csv', 'tracks.csv')
        # Optionally remove the now-empty fma_metadata directory
        try:
            os.rmdir('fma_metadata')
        except OSError:
            pass
        print('Extracted tracks.csv')

# Load FMA track metadata
tracks = pd.read_csv('tracks.csv', header=[0, 1], index_col=0, low_memory=False)
try:
    genre_map = tracks[('track', 'genre_top')].to_dict()
except KeyError:
    print('Available columns:', list(tracks.columns))
    raise

# Prepare a title map for song names
try:
    title_map = tracks[('track', 'title')].to_dict()
except KeyError:
    print('Available columns:', list(tracks.columns))
    raise

MIN_FRAMES = 1000
num_good = 0
num_bad = 0

entries = []
for file in os.listdir(SPECTROGRAM_DIR):
    if file.endswith('.npy'):
        song_id = os.path.splitext(file)[0]
        path = os.path.join(SPECTROGRAM_DIR, file)
        spec = np.load(path)
        if spec.shape[-1] >= MIN_FRAMES:
            genre = genre_map.get(int(song_id), 'Unknown')
            title = title_map.get(int(song_id), 'Unknown')
            entries.append({'song_id': song_id, 'spectrogram_path': path, 'genre': genre, 'title': title})
            num_good += 1
        else:
            num_bad += 1

# Sort by song_id for consistency
entries = sorted(entries, key=lambda x: int(x['song_id']))

# Print first 10 songs and their genres for inspection
print("First 10 songs and their genres:")
for entry in entries[:10]:
    print(f"song_id: {entry['song_id']}, title: {entry['title']}, genre: {entry['genre']}, path: {entry['spectrogram_path']}")

print(f"Songs with at least {MIN_FRAMES} frames: {num_good}")
print(f"Songs with fewer than {MIN_FRAMES} frames: {num_bad}")

pd.DataFrame(entries).to_csv(METADATA_CSV, index=False)
print(f"Saved metadata to {METADATA_CSV} with {len(entries)} entries including genre and title.")

# Extract a simple CSV for manual viewing
simple = tracks[('track', 'title')].to_frame()
simple['genre_top'] = tracks[('track', 'genre_top')]
simple.index.name = 'track_id'
simple.to_csv('tracks_simple.csv')
print("Saved tracks_simple.csv with columns: track_id, title, genre_top") 