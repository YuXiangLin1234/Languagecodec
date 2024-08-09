import os
from datasets import load_dataset
from pydub import AudioSegment
import soundfile as sf
# Load the Common Voice dataset for Taiwanese Mandarin (zh-TW)
dataset = load_dataset("mozilla-foundation/common_voice_16_0", "zh-TW", trust_remote_code=True)

# Create a directory to store the converted audio files
output_dir = "backup/common_voice_zh-TW_wav"
os.makedirs(output_dir, exist_ok=True)

# Function to save and convert audio files
def save_and_convert_audio(example, idx):
	# Temporary path for the .mp3 file
	temp_mp3_path = os.path.join(output_dir, f"{idx}.mp3")
	
	# Write the mp3 file
	sf.write(temp_mp3_path, example['audio']["array"], samplerate=example['audio']['sampling_rate']) 
	
	# Load the mp3 file and convert to wav
	# audio = AudioSegment.from_mp3(temp_mp3_path)
	
	# Path for the .wav file
	# wav_path = os.path.join(output_dir, f"{idx}.wav")
	# audio.export(wav_path, format="wav")

	# Optionally, delete the temporary mp3 file
	# os.remove(temp_mp3_path)
	
	# return wav_path
	return temp_mp3_path

# Download, convert, and save all audio files from the dataset
with open("data/common_voice.txt", "w") as f: 
	for split in ["train", "test", "validation", "other", "invalidated"]:
		for i, example in enumerate(dataset[split]):  # You can also download from other splits like "test", "validation"
			wav_path = save_and_convert_audio(example, i)
			print(wav_path, file = f)
			print(f"Saved {wav_path}")
	

print("Download and conversion completed.")
