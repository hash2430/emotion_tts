import os
file_lists = ['filelists/libritts_selvas_multi_eval.txt',
              'filelists/libritts_selvas_multi_test.txt',
              'filelists/libritts_selvas_multi_train.txt']


for file_list in file_lists:
    miss_cnt = 0
    files = []
    verified_lines = []
    with open(file_list, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        files.append(line.split('|')[0])

    for i in range(len(files)):
        if os.path.exists(files[i]):
            verified_lines.append(lines[i])
        else:
            miss_cnt += 1
    new_file_list = file_list.replace('libritts', 'verified_libritts')
    with open(new_file_list, 'w', encoding='utf-8') as f:
        for line in verified_lines:
            f.write(line)

    print(new_file_list + "has been verified with {} missing counts".format(miss_cnt))


