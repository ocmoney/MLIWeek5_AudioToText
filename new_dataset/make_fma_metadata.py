import os
import pandas as pd

SPECTROGRAM_DIR = 'fma_spectrograms'
METADATA_CSV = 'fma_metadata.csv'

entries = []
for file in os.listdir(SPECTROGRAM_DIR):
    if file.endswith('.npy'):
        song_id = os.path.splitext(file)[0]
        path = os.path.join(SPECTROGRAM_DIR, file)
        entries.append({'song_id': song_id, 'spectrogram_path': path})

# Sort by song_id for consistency
entries = sorted(entries, key=lambda x: int(x['song_id']))

pd.DataFrame(entries).to_csv(METADATA_CSV, index=False)
print(f"Saved metadata to {METADATA_CSV} with {len(entries)} entries.") 