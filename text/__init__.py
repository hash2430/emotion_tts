""" from https://github.com/keithito/tacotron """
import re
from hangul.text import cleaners
from text.symbols import symbols
from text.korean import char_to_id

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


def get_arpabet(word, dictionary):
  word_arpabet = dictionary.lookup(word)
  if word_arpabet is not None:
    return "{" + word_arpabet[0] + "}"
  else:
    return word


def text_to_sequence(text, cleaner_names, lang_code, dictionary=None):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
      dictionary: arpabet class with arpabet dictionary

    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence2 = []


  # Check for curly braces and treat their contents as ARPAbet:
  if lang_code == 0:
    space = _symbols_to_sequence(' ')
    while len(text):
      m = _curly_re.match(text)
      if not m:
        clean_text = _clean_text(text, cleaner_names)
        if cmudict is not None:
          clean_text = [get_arpabet(w, dictionary) for w in clean_text.split(" ")]
          for i in range(len(clean_text)):
              t = clean_text[i]
              if t.startswith("{"):
                sequence2 += _arpabet_to_sequence(t[1:-1])
              else:
                sequence2 +=  _symbols_to_sequence(t)
              sequence2 += space
        else:
          sequence2 += _symbols_to_sequence(clean_text)
        break

      clean_text = _clean_text(text, cleaner_names)
      sequence2 += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
      sequence2 += _arpabet_to_sequence(m.group(2))
      text = m.group(3)
    # remove trailing space
    sequence2 = sequence2[:-1] if sequence2[-1] == space[0] else sequence2

  elif lang_code == 1:
      sequence1 = _clean_text(text, cleaner_names[:-1])
      sequence2, sequence3 = _clean_text(sequence1, [cleaner_names[-1]])

      # remove trailing space
      # space = char_to_id[' ']
      # sequence2 = sequence2[:-1] if sequence2[-1] == space else sequence2
      '''
      sequence1: normalized text
      sequence2: jamo list
      sequence3: jamo index list
      '''
  return sequence1, sequence2, sequence3


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      # Enclose ARPAbet back in curly braces:
      if len(s) > 1 and s[0] == '@':
        s = '{%s}' % s[1:]
      result += s
  return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text


def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
  return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
  return s in _symbol_to_id and s is not '_' and s is not '~'

