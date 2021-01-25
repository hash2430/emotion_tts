import sys
sys.path.append('waveglow/')

from itertools import cycle
import numpy as np
from scipy.io.wavfile import write
import pandas as pd
import librosa
import torch
from torch.utils.data import DataLoader
from model import parse_batch
from configs.two_way_0730 import create_hparams
from train import initiate_model
from waveglow.denoiser import Denoiser
from layers import TacotronSTFT
from data_utils import TextMelLoader, TextMelCollate
from text import cmudict, text_to_sequence
from mellotron_utils import get_data_from_musicxml

hparams = create_hparams()
hparams.batch_size = 1
stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length,
                    hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                    hparams.mel_fmax)
speaker = "nes"
checkpoint_path = '/mnt/sdd1/backup_149/checkpoints/supervised/checkpoint_180000'
model = initiate_model(hparams).cuda().eval()
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
waveglow_path = '/home/admin/projects/mellotron_init_with_single/models/waveglow_256channels_v4.pt'
waveglow = torch.load(waveglow_path)['model'].cuda().eval()
denoiser = Denoiser(waveglow).cuda().eval()
arpabet_dict = cmudict.CMUDict('data/cmu_dictionary')
test_text_path = 'filelists/emotion/emotion_test_grapheme.txt'
test_set = TextMelLoader(test_text_path, hparams)
datacollate = TextMelCollate(1)
dataloader = DataLoader(test_set, num_workers=1, shuffle=False,batch_size=1, pin_memory=False,
                        drop_last=False, collate_fn = datacollate)
speaker_ids = TextMelLoader(hparams.training_files, hparams).speaker_ids
speaker_id = torch.LongTensor([speaker_ids[speaker]]).cuda()

for i, batch in enumerate(dataloader):
    reference_speaker = test_set.audiopaths_and_text[i][2]
    # x: (text_padded, input_lengths, mel_padded, max_len,
    #                  output_lengths, speaker_ids, f0_padded, input_ids, attention_mask),
    # y: (mel_padded, gate_padded)
    x, y = parse_batch(batch)
    x = [x[i]for i in range(len(x))]
    x[5] = speaker_id

    # inputs = text, style_input, speaker_ids, f0s
    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model.inference(x)


    with torch.no_grad():
        audio = denoiser(waveglow.infer(mel_outputs_postnet, sigma=0.8), 0.01)[:, 0]
        audio = audio.squeeze(1).cpu().numpy()
        top_db=25
        for j in range(len(audio)):
            wav, _ = librosa.effects.trim(audio[j], top_db=top_db, frame_length=2048, hop_length=512)
            path="/mnt/sdd1/backup_149/graduate/two-way/gen_test_after_backup/sample-{:03d}_target_speaker-{}_referende_speaker-{}-180000.wav".format(i*hparams.batch_size+j, speaker, reference_speaker)
            write(path, hparams.sampling_rate, wav)
            print('Writing-- '+path)