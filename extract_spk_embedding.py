import shutil
import warnings
import argparse
import torch
import os
import yaml
import fnmatch
import librosa
import numpy as np

from tqdm import tqdm
import pickle
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.inference.ASR import EncoderDecoderASR

def find_audio_files(directory):
    # Define audio file patterns
    audio_patterns = ['*.mp3', '*.wav', '*.flac', '*.aac', '*.ogg']
    
    # List to store paths of found audio files
    audio_files = []
    
    # Walk through the directory tree
    for root, dirs, files in os.walk(directory):
        for pattern in audio_patterns:
            for filename in fnmatch.filter(files, pattern):
                audio_files.append(os.path.join(root, filename))
    
    return audio_files
import torchaudio

def load_and_process_audio(file_path, sr=16000, max_length=None):
    waveform, sample_rate = torchaudio.load(file_path)
    waveform = waveform.mean(dim=0)  # Convert to mono if stereo
    if sample_rate != sr:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sr)(waveform)
    if max_length is not None:
        if waveform.size(0) > max_length:
            waveform = waveform[:max_length]
        elif waveform.size(0) < max_length:
            waveform = torch.nn.functional.pad(waveform, (0, max_length - waveform.size(0)))
    return waveform

def create_audio_tensor(file_paths, sr=16000, max_length=100000):
    processed_audios = [load_and_process_audio(file, sr=sr, max_length=max_length) for file in file_paths]
    audio_tensor = torch.stack(processed_audios)
    return audio_tensor

# def load_and_process_audio(file_path, sr=16000, max_length=None):
#     y, _ = librosa.load(file_path, sr=sr)
#     if max_length is not None:
#         if len(y) > max_length:
#             y = y[:max_length]
#         elif len(y) < max_length:
#             y = np.pad(y, (0, max_length - len(y)))
#     return y

# def create_audio_tensor(file_paths, sr=16000, max_length=100000):

#     # Process each audio to ensure they all have the same length
#     processed_audios = [load_and_process_audio(file, sr=sr, max_length=max_length) for file in file_paths]
    
#     # Convert to NumPy array
#     audio_array = np.array(processed_audios)
    
#     # Convert to PyTorch tensor
#     audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
    
#     return audio_tensor


def main(args):
	classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cuda"})
	files = find_audio_files(args.audio_dir)
	if "cv" in args.audio_dir:
		files = files[:10000]
	# audios = [librosa.load(file, sr=16000)[0] for file in files]
	# max_length = max(len(audio) for audio in audios)

	max_length = 32768
    
	embeddings = {}
	batch_size = 1024
	for i in tqdm(range(0, len(files), batch_size)):
		# n = os.path.join(args.audio_dir, files[i])
		# signal, fs = librosa.load(n, sr = 16000)		
		audios = create_audio_tensor(files[i : i + batch_size], max_length=max_length)
		audios = audios.to("cuda")
		names = files[i: i + batch_size]
			
		embedding = classifier.encode_batch(audios)
		embedding = embedding.cpu().numpy()
		# [batch_size, 1, 192]
		# print(embedding.shape)
          
		for j, e in enumerate(embedding):
			embeddings[names[j]] = e

		torch.cuda.empty_cache()  # Explicitly free GPU memory

	with open(f"{args.output_name}-ecapa-tcnn.pkl", "wb") as f: 
		pickle.dump(embeddings, f)
                  
# python3 reconstruct_superb.py --syn-path syn_path/content_zero -at zero -c
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, default="/work/yuxiang1234/backup/cv-corpus-18.0-2024-06-14/zh-TW/clips")
    parser.add_argument("--output_name", type=str, default="common_voice")
    # parser.add_argument("--audio_dir", type=str, default="/work/yuxiang1234/backup/ML2021_HungyiLee_Corpus")
    # parser.add_argument("--output_name", type=str, default="hy")
    
    args = parser.parse_args()
    main(args)
