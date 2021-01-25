from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
from util import audio
from configs.hparams import hparams


def build_from_path(in_dir, base_dir, out_dir, hparams, num_workers=1, tqdm=lambda x: x):
  '''Preprocesses the LJ Speech dataset from a given input path into a given output directory.

    Args:
      in_dir: The directory where you have downloaded the LJ Speech dataset
      out_dir: The directory to write the output into
      num_workers: Optional number of worker processes to parallelize across
      tqdm: You can optionally pass tqdm to get a nice progress bar

    Returns:
      A list of tuples describing the training examples. This should be written to train.txt
  '''

  # We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
  # can omit it and just call _process_utterance on each input if you want.
  train_dir = os.path.join(base_dir, 'training_{}'.format(out_dir))
  valid_dir = os.path.join(base_dir, 'valid_{}'.format(out_dir))
  os.makedirs(train_dir, exist_ok=True)
  os.makedirs(valid_dir, exist_ok=True)
  executor = ProcessPoolExecutor(max_workers=num_workers)
  futures = []
  futures_val = []
  # index = 1
  with open(os.path.join(in_dir, 'script_punc_num.txt'), encoding='cp949', errors='ignore') as f:
    for line in f:
      parts = line[0:5]  # line.strip().split('|')
      index = line[1:5]
      wav_path = os.path.join(in_dir, 'wav', 'kaist_sssc_korean_%s.wav' % parts)  # [0])
      test_list = ['4030', '4048', '4053', '4078', '4089', '4092', '4093', '4096', '4117', '4120', '4122', '4130', '4142', '4156', '4201', '4213', '4227', '4309', '4319', '4323']
      if os.path.exists(wav_path):
        if int(index)>4000 and not (index in test_list):
          text = line[6:-1]  # parts[2]
          futures_val.append(executor.submit(partial(_process_utterance, valid_dir, index, wav_path, text)))
        elif int(index)<=4000:
          text = line[6:-1]  # parts[2]
          futures.append(executor.submit(partial(_process_utterance, train_dir, index, wav_path, text)))
      # index += 1
  # return [future.result() for future in tqdm(futures)]
    write_metadata([future.result() for future in tqdm(futures)], train_dir, 'train.txt')
    write_metadata([future.result() for future in tqdm(futures_val)], valid_dir, 'valid.txt')


def _process_utterance(out_dir, index, wav_path, text):
  wav = audio.load_wav(wav_path)
  spectrogram = audio.spectrogram(wav).astype(np.float32)
  n_frames = spectrogram.shape[1]
  mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
  spectrogram_filename = 'sitec-spec-%04d.npy' % int(index)
  mel_filename = 'sitec-mel-%04d.npy' % int(index)
  np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
  np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
  return (spectrogram_filename, mel_filename, n_frames, text)


def write_metadata(metadata, out_dir, domain):
  with open(os.path.join(out_dir, domain), 'w', encoding='utf-8') as f:
    for m in metadata:
      f.write('|'.join([str(x) for x in m]) + '\n')
  frames = sum([m[2] for m in metadata])
  hours = frames * hparams.frame_shift_ms / (3600 * 1000)
  print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))
  print('Max input length:  %d' % max(len(m[3]) for m in metadata))
  print('Max output length: %d' % max(m[2] for m in metadata))