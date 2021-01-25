def modify_phoneme_script_to_create_grapheme_script(original_dataset_path, grapheme_dataset_path):

    with open(original_dataset_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        split_result = line.split('|')
        wav_path = split_result[0]
        speaker = split_result[2].rstrip()
        speaking_emotion = 0
        content_emotion = 0
        txt_path = wav_path.replace('selvas_wav', 'selvas_txt').replace('wav_trimmed_22050', 'script').replace('.wav',
                                                                                                               '.txt')
        with open(txt_path, 'r', encoding='utf-8-sig') as f:
            txt = f.readline().rstrip()
        # new_line = '{}|{}|{}|{}|{}'.format(wav_path,txt, speaker, speaking_emotion, content_emotion)
        new_line = txt
        new_lines.append(new_line)

    with open(grapheme_dataset_path, 'w', encoding='utf-8') as f:
        for line in new_lines:
            f.write(line+'\n')

if __name__ == '__main__':
    # original_dataset_path = '/home/admin/projects/graduate/emotion_vector/filelists/selvas_main_train.txt'
    # grapheme_dataset_path = '/home/admin/projects/graduate/emotion_vector/filelists/grapheme/grapheme_selvas_main_train_tmp.txt'
    original_dataset_path = '/home/admin/projects/graduate/emotion_vector/filelists/single_language_selvas/train_file_list_pron.txt'
    grapheme_dataset_path = '/home/admin/projects/graduate/emotion_vector/filelists/grapheme_selvas_multi_train_tmp.txt'
    modify_phoneme_script_to_create_grapheme_script(original_dataset_path, grapheme_dataset_path)