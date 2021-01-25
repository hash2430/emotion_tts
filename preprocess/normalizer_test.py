from text.cleaners import transliteration_cleaners, korean_cleaners
# text_file='/home/admin/projects/graduate/emotion_vector/filelists/grapheme/grapheme_selvas_main_train_tmp.txt'
text_file='/home/admin/projects/graduate/emotion_vector/filelists/grapheme/grapheme_selvas_multi_train_tmp.txt'
i=1
with open(text_file, 'r', encoding='utf-8') as f:
    texts = f.readlines()
for text in texts:
    try:
        seq1 = transliteration_cleaners(text)
        seq2 = korean_cleaners(seq1)
    except:
        print(text)
        i+=1