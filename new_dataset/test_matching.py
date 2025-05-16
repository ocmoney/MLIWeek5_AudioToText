import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from model import SongEmbeddingNet
import random
from tqdm import tqdm
import os

def load_model(model_path, device):
    model = SongEmbeddingNet(temperature=0.07, normalize_embeddings=True).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_track_info(metadata_file):
    """Load track information from fma_metadata.csv"""
    # Read the CSV file
    metadata_df = pd.read_csv(metadata_file)
    
    # Create a mapping from track ID to genre
    track_to_genre = {}
    
    # Create the mapping
    for _, row in metadata_df.iterrows():
        track_id = int(row['song_id'])
        track_to_genre[track_id] = row['genre']
    
    return track_to_genre

def get_spectrogram_paths(metadata_file):
    """Get all spectrogram file paths from metadata"""
    metadata_df = pd.read_csv(metadata_file)
    return metadata_df['spectrogram_path'].tolist()

def get_random_slice(spectrogram_path, slice_frames=216):
    """Get a random 5-second slice from a spectrogram"""
    spec = np.load(spectrogram_path)
    if spec.ndim == 2:
        spec = np.expand_dims(spec, 0)
    
    time_dim = spec.shape[-1]
    max_start = time_dim - slice_frames
    start_idx = np.random.randint(0, max_start)
    return spec[..., start_idx:start_idx+slice_frames]

def get_all_embeddings(model, spectrogram_paths, device, batch_size=32):
    """Get embeddings for all spectrograms"""
    all_embeddings = []
    all_track_ids = []
    
    # Process in batches
    for i in tqdm(range(0, len(spectrogram_paths), batch_size), desc="Computing embeddings"):
        batch_paths = spectrogram_paths[i:i+batch_size]
        batch_specs = []
        
        # Load and process each spectrogram in the batch
        for path in batch_paths:
            spec = np.load(path)
            if spec.ndim == 2:
                spec = np.expand_dims(spec, 0)
            # Take the first 5 seconds
            spec = spec[..., :216]  # 216 frames = 5 seconds
            batch_specs.append(spec)
        
        # Convert to tensor
        batch_tensor = torch.from_numpy(np.stack(batch_specs)).float().to(device)
        
        # Get embeddings
        with torch.no_grad():
            embeddings = model(batch_tensor)
        
        all_embeddings.append(embeddings.cpu())
        
        # Get track IDs from filenames
        track_ids = [int(os.path.basename(p).split('.')[0]) for p in batch_paths]
        all_track_ids.extend(track_ids)
    
    return torch.cat(all_embeddings, dim=0), all_track_ids

def find_matching_song(query_embedding, all_embeddings, all_track_ids, track_to_genre, top_k=5):
    """Find the top-k closest songs to the query embedding"""
    # Move all_embeddings to the same device as query_embedding
    all_embeddings = all_embeddings.to(query_embedding.device)
    
    # Calculate cosine similarity (since embeddings are normalized)
    similarities = torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), all_embeddings)
    
    # Get top-k most similar songs
    top_k_similarities, top_k_indices = torch.topk(similarities, k=top_k, largest=True)
    
    # Convert tensors to lists for easier handling
    similarities_list = top_k_similarities.cpu().tolist()
    indices_list = top_k_indices.cpu().tolist()
    
    # Get corresponding track IDs and genres
    results = []
    for i in range(top_k):
        idx = indices_list[i]
        sim = similarities_list[i]
        track_id = all_track_ids[idx]  # Now idx is a single integer
        genre = track_to_genre.get(track_id, "Unknown")
        results.append({
            'track_id': track_id,
            'genre': genre,
            'similarity': sim
        })
    
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = load_model("song_classifier.pth", device)
    
    # Load track information
    print("Loading track information...")
    track_to_genre = load_track_info("fma_metadata.csv")
    print(f"Loaded {len(track_to_genre)} tracks with genre information")
    
    # Get spectrogram paths
    spectrogram_paths = get_spectrogram_paths("fma_metadata.csv")
    print(f"Found {len(spectrogram_paths)} spectrograms")
    
    # Get embeddings for all songs
    print("Computing embeddings for all songs...")
    all_embeddings, all_track_ids = get_all_embeddings(model, spectrogram_paths, device)
    
    # Test genre prediction
    num_tests = 10
    print(f"\nRunning {num_tests} genre prediction tests...")
    
    for test_idx in range(num_tests):
        # Pick a random spectrogram
        random_path = random.choice(spectrogram_paths)
        random_track_id = int(os.path.basename(random_path).split('.')[0])
        true_genre = track_to_genre.get(random_track_id, "Unknown")
        
        # Get a random slice
        query_slice = get_random_slice(random_path)
        query_slice = torch.from_numpy(query_slice).float().unsqueeze(0).to(device)
        
        # Get embedding for query slice
        with torch.no_grad():
            query_embedding = model(query_slice)
        
        # Find matching songs
        matches = find_matching_song(query_embedding, all_embeddings, all_track_ids, track_to_genre)
        
        # Print results
        print(f"\nTest {test_idx + 1}:")
        print(f"Track ID: {random_track_id}")
        print(f"True genre: {true_genre}")
        print("\nTop 5 matches:")
        for i, match in enumerate(matches):
            print(f"{i+1}. Track {match['track_id']} - {match['genre']} (similarity: {match['similarity']:.3f})")
        print()

if __name__ == "__main__":
    main() 