
from languagecodec_encoder.utils import convert_audio
import torchaudio
import torch
from languagecodec_decoder.pretrained import Vocos
from pydub import AudioSegment

device=torch.device('cuda')

config_path = "/work/yuxiang1234/Languagecodec/configs/languagecodec_mm.yaml"
model_path = "/work/yuxiang1234/backup/languagecodec_paper.ckpt"
languagecodec = Vocos.from_pretrained0802(config_path, model_path)
languagecodec = languagecodec.to(device)
audios = ["/work/yuxiang1234/Languagecodec/audio-for-tps/esc50.wav", "/work/yuxiang1234/Languagecodec/audio-for-tps/librispeech.wav", "/work/yuxiang1234/Languagecodec/audio-for-tps/urbansound8k.wav"]

for audio_path in audios:
	wav, sr = torchaudio.load(audio_path)
	wav = convert_audio(wav, sr, 24000, 1) 
	bandwidth_id = torch.tensor([0])
	wav = wav.to(device)
	_,discrete_code= languagecodec.encode_infer(wav, bandwidth_id=bandwidth_id)
	print(discrete_code)
	
	audio = AudioSegment.from_file(audio_path)

	# Get the length in milliseconds
	length_in_milliseconds = len(audio)

	# Convert to seconds
	length_in_seconds = length_in_milliseconds / 1000

	print(f"Length: {length_in_seconds} seconds")