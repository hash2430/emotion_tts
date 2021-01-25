from configs.hparams import create_hparams
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import librosa
from utils import read_wav_np
import os, random
hparams = create_hparams()
from scipy.io.wavfile import write
import torch
def build_from_path(in_dir, out_dir, filelist_names, spk_name_idx,num_workers=16, tqdm=lambda x: x):
    wav_files = []
    # for all speakers, count index and either add to train_list/eval_list/test_list
    speakers = os.listdir(in_dir)
    for speaker in speakers:
        path = os.path.join(in_dir, speaker, 'wav_22050')
        wavs = os.listdir(path)
        for wav in wavs:
            wav_files.append(os.path.join(path, wav))
    random.shuffle(wav_files)
    # wav_files.sort()
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    for wav_file in wav_files:
        futures.append(executor.submit(partial(_process_utterance, wav_file, spk_name_idx)))
    write_metadata([future.result() for future in tqdm(futures)], out_dir, filelist_names[0])
'''
1. Read each file
2. Down sample to 22050Hz
3. Create meta file with the format 'path|phonemes|speaker'
["g",  "n",  "d",  "l",  "m",  "b",  "s",  "-", "j",   "q",
                "k", "t", "p", "h", "x", "w", "f", "c", "z", "A",
                "o", "O", "U", "u", "E", "a", "e", "1", "2", "3",
                "4", "5", "6", "7", "8", "9", "[", "]", "<", ">",
                "G", "N", "D", "L", "M", "B", "0", "K", ";;",";", "sp", "*",
                "$", "?", "!","#"]
'''
# I have not decided whether to separate dierectories for train/eval/test
def _process_utterance(in_path, spk_name_idx):
    # wav is saved as int 16
    # int16 is converted into float32 here
    txt_file = in_path.replace('wav_22050', 'script').replace('.wav', '.txt')
    with open(txt_file, 'r', encoding='utf-8-sig') as f:
        line = f.readline().rstrip()

    speaker = in_path.split('/')[spk_name_idx]
    file_name = os.path.basename(in_path)
    emotion = int(file_name[5:8])
    if emotion <101:
        emotion_label = 0
    elif emotion < 201:
        emotion_label = 1
    elif emotion < 301:
        emotion_label = 2
    else:
        emotion_label = 3
    return (in_path, line.rstrip('\n'), speaker, emotion_label, emotion_label)



def write_metadata(metadata, out_dir, out_file):
    with open(os.path.join(out_dir, out_file), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) +'\n')