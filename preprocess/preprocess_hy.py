import numpy as np
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset

from scipy.io.wavfile import write
ds = load_dataset('chiyuanhsiao/ML2021_HungyiLee_Corpus', split='test')[:19291]

# Duration - 41818.9s (11.6h)
# duration = 0.0
# for _, audio, _ in zip(ds['file'], ds['audio'], ds['transcription']):
#     leng = audio['array'].shape[0]
#     sr = audio['sampling_rate']
#     if sr != 16000:
#         print(sr)
#     duration += 1.0 * leng / sr

# print(duration)

# print(ds['audio'][0])

p_save = Path('/work/yuxiang1234/backup/ML2021_HungyiLee_Corpus')
p_save.mkdir(parents=True, exist_ok=True)

for name, audio, _ in tqdm(zip(ds['file'], ds['audio'], ds['transcription'])):
    p_wav = p_save / name
    audio_array_int16 = np.int16(audio['array'] * 32767)
    write(str(p_wav), audio['sampling_rate'], audio_array_int16)
