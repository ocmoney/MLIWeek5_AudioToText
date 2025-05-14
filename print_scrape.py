import os
from pydub import AudioSegment
from pydub.playback import play

DOWNLOAD_DIR = "fma_hiphop_downloads"

def list_audio_files(base_dir):
    audio_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith((".mp3", ".wav")):
                audio_files.append(os.path.join(root, file))
    return audio_files

def main():
    print("🎧 Scanning for songs...\n")
    audio_files = list_audio_files(DOWNLOAD_DIR)

    if not audio_files:
        print("No audio files found.")
        return

    for i, path in enumerate(audio_files):
        print(f"{i+1}. {os.path.basename(path)}")

    try:
        choice = int(input("\nSelect a song to play (1–{}): ".format(len(audio_files))))
        if 1 <= choice <= len(audio_files):
            file_path = audio_files[choice - 1]
            print(f"\n▶️ Playing: {os.path.basename(file_path)}")
            song = AudioSegment.from_file(file_path)
            play(song)
        else:
            print("Invalid choice.")
    except ValueError:
        print("Invalid input.")

if __name__ == "__main__":
    main()
