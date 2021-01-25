import os
import shutil

reference_list_path = 'subjective_test/subjective_script4_emotive.txt'
store_path = '/mnt/sdc1/mellotron_experiment/subjective_reference4_emotive/'
with open(reference_list_path, 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()
references = []
for i in range(len(lines)):
    reference = lines[i].split('|')[0]
    speaker = lines[i].split('|')[-2]
    name = 'sample-{:03d}_refer-{}.wav'.format(i, speaker)
    save_path = os.path.join(store_path, name)
    shutil.copy(reference, save_path)