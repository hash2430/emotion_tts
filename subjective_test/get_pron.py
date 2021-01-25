import os
script = '/home/admin/projects/mellotron_grl/subjective_test/emotive2_tmp.txt'
with open(script, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_paths = []
for line in lines:
    new_line = line.rstrip()
    pron_path = new_line.replace('wav_22050', 'script').replace('.wav', '.pron')
    with open(pron_path, 'r', encoding='utf-8-sig') as f:
        pron = f.readline().rstrip()
    new_line = new_line + '|'+pron+'|neb'+'|1'
    new_paths.append(new_line)

new_script = '/home/admin/projects/mellotron_grl/subjective_test/emotive2.txt'
with open(new_script, 'w', encoding='utf-8') as f:
    for line in new_paths:
        f.write(line + '\n')
