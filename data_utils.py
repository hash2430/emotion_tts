import random
import os
import re
import numpy as np
import torch
import torch.utils.data
import librosa

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cmudict
from yin import compute_yin
from nlp.tokenization_kobert import KoBertTokenizer
from torch.nn.utils.rnn import pad_sequence

class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio, text and speaker ids
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms and f0s from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams, speaker_ids=None, output_directory=None):
        self.tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.f0_min = hparams.f0_min
        self.f0_max = hparams.f0_max
        self.harm_thresh = hparams.harm_thresh
        self.p_arpabet = hparams.p_arpabet

        self.cmudict = None
        if hparams.cmudict_path is not None:
            self.cmudict = cmudict.CMUDict(hparams.cmudict_path)

        self.speaker_ids = speaker_ids
        if speaker_ids is None:
            self.speaker_ids = self.create_speaker_lookup_table(self.audiopaths_and_text)

        # print speaker_lookup_table
        if not (output_directory is None) and not (self.speaker_ids is None):
            speaker_id_path = os.path.join(output_directory, 'speaker_ids.txt')

            with open(speaker_id_path, 'w', encoding='utf-8') as f:
                for key, value in self.speaker_ids.items():
                    f.write('{}: {}\n'.format(key, value))


        # random.seed(1234)
        # random.shuffle(self.audiopaths_and_text)

    def create_speaker_lookup_table(self, audiopaths_and_text):
        speaker_list = [x[2] for x in audiopaths_and_text]
        speaker_ids = np.sort(np.unique(speaker_list))
        d = {speaker_ids[i]: i for i in range(len(speaker_ids))}
        return d

    def get_f0(self, audio, sampling_rate=22050, frame_length=1024,
               hop_length=256, f0_min=100, f0_max=300, harm_thresh=0.1):
        f0, harmonic_rates, argmins, times = compute_yin(
            audio, sampling_rate, frame_length, hop_length, f0_min, f0_max,
            harm_thresh)
        pad = int((frame_length / hop_length) / 2)
        f0 = [0.0] * pad + f0 + [0.0] * pad

        f0 = np.array(f0, dtype=np.float32)
        return f0

    def get_data(self, audiopath_and_text):

        try:
            audiopath, text, speaker, speaking_emotion, text_emotion = audiopath_and_text
            lang_code = 1
            text_nlp, text_tts, text_tts_idx = self.get_text(text, lang_code)
            # TODO:
            # sync text_nlp and text_tts
            # call tokenizer.encode() to obtain input_ids
            tokenized_text = self.tokenizer.tokenize(text_nlp)
            dict = self.tokenizer.encode_plus(text_nlp)



            ###
            mel, f0 = self.get_mel_and_f0(audiopath)
            speaker_id = self.get_speaker_id(speaker)
            return (text_tts_idx, mel, speaker_id, f0, dict['input_ids'], dict['attention_mask'], speaking_emotion)
        except:
            print(audiopath_and_text)

    def get_speaker_id(self, speaker_id):
        return torch.IntTensor([self.speaker_ids[speaker_id]])

    def get_mel_and_f0(self, filepath):
        audio, sampling_rate = load_wav_to_torch(filepath)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        audio_norm = audio / self.max_wav_value # max_wav_value must be set to 1 when wav is float32 format already
        # I changed them to float32 during preprocessing so this normalization is unnecessary.
        audio_norm = audio_norm.unsqueeze(0)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)

        f0 = 0 #dummy

        return melspec, f0

    def get_text(self, text, lang_code):
        text1, text2, text3 = text_to_sequence(text, self.text_cleaners, lang_code, self.cmudict)
        text3= torch.IntTensor(text3)
        return text1, text2, text3

    def __getitem__(self, index):
        return self.get_data(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)

class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded, gate padded and speaker ids
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        speaker_ids = torch.LongTensor(len(batch))
        speaking_emotion = torch.LongTensor(len(batch))

        f0_padded = torch.FloatTensor(len(batch), 1, max_target_len)
        f0_padded.zero_()
        _,_,_,_,input_ids, labels,_ = zip(*batch)
        attention_mask = [[1] * len(input_id) for input_id in input_ids]

        input_ids = pad_sequence([torch.Tensor(input_id).to(torch.long) for input_id in input_ids],
                                 padding_value=0, batch_first=True)
        attention_mask = pad_sequence([torch.Tensor(mask).to(torch.long) for mask in attention_mask],
                                      padding_value=0, batch_first=True)



        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
            speaker_ids[i] = batch[ids_sorted_decreasing[i]][2]
            speaking_emotion[i] = int(batch[ids_sorted_decreasing[i]][6])


        model_inputs = (text_padded, input_lengths, mel_padded, gate_padded,
                        output_lengths, speaker_ids, f0_padded, input_ids, attention_mask, speaking_emotion)

        return model_inputs
