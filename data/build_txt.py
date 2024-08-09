import os
import csv

def find_audio_files_and_write_to_csv(directory, csv_filename, audio_extensions=None):
    if audio_extensions is None:
        audio_extensions = ['.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a', '.wma']  # Add more extensions as needed

    audio_files = []

    # Walk through the directory tree
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file has an audio extension
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                # Get the absolute path of the audio file
                abs_path = os.path.abspath(os.path.join(root, file))
                audio_files.append(abs_path)

    # Write the absolute paths to a CSV file
    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for audio_file in audio_files:
            writer.writerow([audio_file])
    
    print(f"Found {len(audio_files)} audio files. Paths written to {csv_filename}")

# Example usage:
directory_to_search = '/home/yxlin/backup/cv-corpus-18.0-2024-06-14/zh-TW/clips'  # Replace with your directory path
output_csv = 'common_voice_18.txt'  # Output CSV file name

find_audio_files_and_write_to_csv(directory_to_search, output_csv)
