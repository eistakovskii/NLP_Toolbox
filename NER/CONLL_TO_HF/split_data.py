ner_tags_to_keep = set(["B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"]) # TAGS TO KEEP

with open(r'PATH_TO_YOU_DATA\all_data_ner.txt', encoding='utf-8') as f: # works just fine with either txt files or conll files
    all_lines = f.readlines()
    num_of_samples = 0
    
    for v in all_lines:
        if v == '\n':
            num_of_samples += 1
    train_split = round(num_of_samples * 0.8)
    val_test_split = round(num_of_samples * 0.2)
    val_split = round(val_test_split/2)
    test_split = round(val_test_split/2)

    num_of_samples_t = 0
    train_split_l = list()
    val_split_l = list()
    test_split_l = list()


    for v in all_lines:
        if v == '\n':
            num_of_samples_t += 1
        temp_str = v.replace(' -X- _ ', ' ')

        if temp_str != '\n' and temp_str.split()[-1] not in ner_tags_to_keep:
            temp_l = temp_str.split()
            temp_l[-1] = 'O'
            temp_str = ' '.join(temp_l) + '\n'

        if num_of_samples_t <= train_split:
            train_split_l.append(temp_str)
        elif num_of_samples_t > train_split and num_of_samples_t <= (train_split + val_split):
            val_split_l.append(temp_str)
        elif num_of_samples_t > (train_split + val_split):
            test_split_l.append(temp_str)


with open(f'train.txt', mode = 'w', encoding = 'utf-8') as f2:
    for i in train_split_l[1:]:
        f2.write(i)

with open(f'valid.txt', mode = 'w', encoding = 'utf-8') as f2:
    for i in val_split_l[1:]:
        f2.write(i)

with open(f'test.txt', mode = 'w', encoding = 'utf-8') as f2:
    for i in test_split_l[1:]:
        f2.write(i)
