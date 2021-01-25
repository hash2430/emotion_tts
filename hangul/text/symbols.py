'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
'''
from configs.hparams import create_hparams  # hparams, hparams_debug_string

from .korean import ALL_SYMBOLS

hparams=create_hparams()
if hparams.text_cleaners != 'korean_cleaners':
  # symbols = '_~ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? '
  print('english cleaner')
  symbols = '_~ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;?<>《》 '
else:
  print('korean cleaner')
  symbols = ALL_SYMBOLS
