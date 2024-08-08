import shutil
import warnings
import argparse
import torch
import os
import yaml
import fnmatch

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

    config_path = "/work/yuxiang1234/Languagecodec/configs/languagecodec_mm.yaml"
    # model_path = "/work/yuxiang1234/Languagecodec/results-3e-4/lightning_logs/qtzj1bap/checkpoints/epoch=9-step=10000.ckpt"
    model_path = "/work/yuxiang1234/backup/languagecodec_paper.ckpt"
    languagecodec = Vocos.from_pretrained0802(config_path, model_path)
    languagecodec = languagecodec.to(device)
    for i, source in enumerate(audio_files):
        if "syn" in source:
            continue
        print(f"{i} / {len(audio_files)} ", source)
        # source_audio = librosa.load(source, sr=24000)[0]

        wav, sr = torchaudio.load(source)
        source_audio = convert_audio(wav, sr, 24000, 1) 
        bandwidth_id = torch.tensor([0]).to(device)
        source_audio = source_audio.to(device)
        features, discrete_code= languagecodec.encode_infer(source_audio, bandwidth_id=bandwidth_id)
        print(features.shape)
        print(discrete_code.shape)
        audio_out = languagecodec.decode(features, bandwidth_id=bandwidth_id) 
        if "ref_path" in source:
            target_name = source.replace("ref_path", args.syn_path)
        else: 
            target_name = source.replace(".wav", "") + "syn.wav"

        save_dir = target_name.replace(os.path.basename(target_name), "")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            # shutil.rmtree(save_dir)
        
        torchaudio.save(target_name, audio_out.cpu(), sample_rate=24000, encoding='PCM_S', bits_per_sample=16)
            
        # torchaudio.save(target_name, full_pred_wave[0].cpu(), 24000)

    copy_files_without_wav(args.ref_path, args.ref_path.replace("ref_path", args.syn_path))

# python3 reconstruct_superb.py --syn-path syn_path/content_zero -at zero -c
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, default="")
    parser.add_argument("--config-path", type=str, default="")
    # parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--ref-path", type=str, default="/work/yuxiang1234/backup/chinese-audio-sample")
    parser.add_argument("--syn-path", type=str, default="lc_syn_path/hy-3e-4")

    args = parser.parse_args()
    main(args)
