def do_the_splits(PATH_TO_YOUR_DATA: str, NER_TAGS_TO_KEEP: list, export_splits = False) -> list:

    """
    DESCRIPTION:
    
    Takes a path to a txt files in the CONLL format (4 columns present) and splits it into 3 splits: train, validation, test. The proportion is 80, 10, 10.
    The data is separated not by lines but rather by sentences/text samples. When exporting from Label Studio you can notice that the text are not only tokenized but also
    split with a newline character into sentences. The splits thus were done by counting the newlines character.
    Note that the sentences are not randomized upon splitting.
    
    ARGS:
    
    PATH_TO_YOUR_DATA: path to your tagged raw data which has 4 columns separated by a space: e.g. "машина -X- _ O"
    NER_TAGS_TO_KEEP: indicate which tags you want to keep, applicable in particular when you have many tags and you need only few
    
    OUTPUT:
    
    Returns 4 lists: train split, validation split, test split, and also a list of tags
    
    """
    
    tag_list_temp = list('O')
    
    for tag in NER_TAGS_TO_KEEP:
        bio_tags_out = [prefix + tag for prefix in ['B-', 'I-']]
        tag_list_temp.extend(bio_tags_out)
    
    ner_tags_to_keep = set(tag_list_temp)

    with open(PATH_TO_YOUR_DATA, encoding='utf-8') as f: # works just fine with either txt files or conll files
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


    if export_splits:
        
        with open(f'train.txt', mode = 'w', encoding = 'utf-8') as f2:
            for i in train_split_l[1:]:
                f2.write(i)

        with open(f'valid.txt', mode = 'w', encoding = 'utf-8') as f2:
            for i in val_split_l[1:]:
                f2.write(i)

        with open(f'test.txt', mode = 'w', encoding = 'utf-8') as f2:
            for i in test_split_l[1:]:
                f2.write(i)

    train_split_out = train_split_l[1:]
    val_split_out = val_split_l[1:]
    test_split_out = test_split_l[1:]
    
    return train_split_out, val_split_out, test_split_out, ner_tags_to_keep
