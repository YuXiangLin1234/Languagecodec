import shutil
import warnings
import argparse
import torch
import os
import yaml
import fnmatch
import numpy as np
from collections import defaultdict
import time
from pydub import AudioSegment

def get_audio_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    duration_in_ms = len(audio)
    duration_in_seconds = duration_in_ms / 1000.0
    return duration_in_seconds


import os
os.environ['TRANSFORMERS_CACHE'] = '/work/yuxiang1234/cache'
os.environ['HF_DATASETS_CACHE']="/work/yuxiang1234/cache"
# os.environ["HF_HOME"] = "/work/yuxiang1234/cache"

warnings.simplefilter('ignore')


from languagecodec_encoder.utils import convert_audio
import torchaudio
import torch
from languagecodec_decoder.pretrained import Vocos

import torchaudio
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def copy_files_without_wav(src_dir, dest_dir):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Walk through the source directory
    for root, dirs, files in os.walk(src_dir):
        # Create corresponding subdirectories in the destination directory
        for dir_name in dirs:
            src_subdir = os.path.join(root, dir_name)
            dest_subdir = os.path.join(dest_dir, os.path.relpath(src_subdir, src_dir))
            if not os.path.exists(dest_subdir):
                os.makedirs(dest_subdir)
        
        # Copy files, excluding .wav files
        for file_name in files:
            if not file_name.lower().endswith('.wav') and not file_name.lower().endswith('.mp3') and not file_name.lower().endswith('.flac') and not file_name.lower().endswith('.aac') and not file_name.lower().endswith('.ogg'):
                src_file = os.path.join(root, file_name)
                dest_file = os.path.join(dest_dir, os.path.relpath(src_file, src_dir))
                dest_file_dir = os.path.dirname(dest_file)
                
                # Create the directory if it doesn't exist
                if not os.path.exists(dest_file_dir):
                    os.makedirs(dest_file_dir)
                
                shutil.copy2(src_file, dest_file)
                print(f"Copied {src_file} to {dest_file}")

@torch.no_grad()
def main(args):

    audio_files = find_audio_files(args.ref_path)
    all_ckpts = ["/home/yxlin/Languagecodec/pretrained_models/languagecodec_paper.ckpt"]
    config_path = "/home/yxlin/Languagecodec/configs/languagecodec_mm.yaml"
    
    save_dir = "/home/yxlin/backup/codec-infer/language-codec"
    log_file = os.path.join(save_dir, "log.txt")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    f = open(log_file, "w")

    encoder_time = []
    decoder_time = []
    wav_time = []

    audio_files  = [
    "/home/yxlin/backup/chinese-audio-sample/_ML Lecture 1_ Regression - Demo.mp3",
                 "/home/yxlin/backup/chinese-audio-sample/_fu-xuan.wav",
                 "/home/yxlin/backup/chinese-audio-sample/_hook.wav",
                 "/home/yxlin/backup/chinese-audio-sample/_old oti.wav",
                 "/home/yxlin/backup/chinese-audio-sample/audio1.mp3",
                 "/home/yxlin/backup/chinese-audio-sample/audio2.mp3",
                 "/home/yxlin/backup/chinese-audio-sample/audio3.mp3"]
    reconstruct_audios = defaultdict(list)
    for model_path in all_ckpts:
        languagecodec = Vocos.from_pretrained0802(config_path, model_path)
        languagecodec = languagecodec.to(device)
        for i, source in enumerate(audio_files):

            encoder_start_time = 0.0
            decoder_start_time = 0.0
            encoder_end_time = 0.0
            decoder_end_time = 0.0

            print(f"{i} / {len(audio_files)} ", source)
            # source_audio = librosa.load(source, sr=24000)[0]

            wav, sr = torchaudio.load(source)
            source_audio = convert_audio(wav, sr, 24000, 1) 
            bandwidth_id = torch.tensor([0]).to(device)
            source_audio = source_audio.to(device)

            encoder_start_time = time.perf_counter()
            features, discrete_code = languagecodec.encode_infer(source_audio, bandwidth_id=bandwidth_id)
            encoder_end_time = time.perf_counter()

            print(features.shape)
            print(discrete_code.shape)

            decoder_start_time = time.perf_counter()
            audio_out = languagecodec.decode(features, bandwidth_id=bandwidth_id) 
            decoder_end_time = time.perf_counter()

            audio_out = audio_out.cpu()
            reconstruct_audios[i].append(audio_out)
            
            torchaudio.save(os.path.join(save_dir, os.path.basename(source)), audio_out.cpu(), sample_rate=24000, encoding='PCM_S', bits_per_sample=16)

            w = get_audio_duration(source)
            wav_time.append(w)
            e = encoder_end_time - encoder_start_time
            d = decoder_end_time - decoder_start_time
            encoder_time.append(e)
            decoder_time.append(d)
            # torchaudio.save(target_name, full_pred_wave[0].cpu(), 24000)
            print(f"\rwav length: {w:.1f} s", file = f)
            print(f"encoder: {e:.1f} s / rtf: {w:.4f} (↑)", file = f)
            print(f"decoder: {d:.1f} s / rtf: {w/d:.4f} (↑)", file = f)       


        print(f"encoder rtf: {sum(wav_time)/sum(encoder_time):.4f} (↑)")
        print(f"decoder rtf: {sum(wav_time)/sum(decoder_time):.4f} (↑)")
        print(f"encoder rtf: {sum(wav_time)/sum(encoder_time):.4f} (↑)", file = f)
        print(f"decoder rtf: {sum(wav_time)/sum(decoder_time):.4f} (↑)", file = f)

    f.close()
    # for key in reconstruct_audios.keys():
    #     print(reconstruct_audios)
    #     print(key, np.mean((reconstruct_audios[key][0] - reconstruct_audios[key][1]) ** 2))
    # copy_files_without_wav(args.ref_path, args.ref_path.replace("ref_path", args.syn_path))

# python3 reconstruct_superb.py --syn-path syn_path/content_zero -at zero -c
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, default="")
    parser.add_argument("--config-path", type=str, default="")
    # parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--ref-path", type=str, default="/home/yxlin/backup/chinese-audio-sample")
    parser.add_argument("--syn-path", type=str, default="lc_syn_path/hy-3e-4")

    args = parser.parse_args()
    main(args)
