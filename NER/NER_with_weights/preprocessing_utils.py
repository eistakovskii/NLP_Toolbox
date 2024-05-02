import datasets

def do_the_preprocessing(PATH_TO_YOUR_DATA: str, TARGET_NER_GROUP: str ) -> list:

    """
    DESCRIPTION:

    Takes a path to a txt files in the CONLL format (4 columns present)

    ARGS:

    PATH_TO_YOUR_DATA: path to your tagged raw data which has N columns separated by a space: e.g. "машина O O O". First column is always tokens
    TARGET_NER_GROUP: indicate which tags group you want to keep, i.e. which column from your file

    OUTPUT:

    Returns all samples unsorted with following structure [[(token, label), (token, label)], []]

    """
    def is_russian(string):
        """
        checks whether the given string is written in Russian or not
        """
        for char in string:
            if ord(char) >= 1040 and ord(char) <= 1103:
                # The character is within the range of Cyrillic characters used in Russian
                return True
        return False

    with open(PATH_TO_YOUR_DATA, encoding='utf-8') as f: # works just fine with either txt files or conll files
        all_lines = f.readlines()

    header = all_lines[0].strip().split()
    header = [i[1:-1] for i in header]

    TARGET_NER_GROUP_INDEX = header.index(TARGET_NER_GROUP)

    all_lines = all_lines[2:-1] # get all the lines except the header

    samples = []
    current_sample = []

    for line in all_lines: # iterate through lines
        if line == '\n': # for us the newline character is the sign of the start or the end of the sample
            if current_sample:
                samples.append(current_sample)
                current_sample = []
        elif len(line.split()) > 5:
            continue
        else:
            temp_str = line.strip()
            temp_l = temp_str.split()

            token = temp_l[0]
            target_ner_group_label = temp_l[TARGET_NER_GROUP_INDEX]
            if is_russian(target_ner_group_label): # the same as below, mainly trying to avoid the regex mistakes
                current_sample.append((token, 'O'))
            elif 'True' in target_ner_group_label: # fixing the regex problems TODO: after fully switching to .conll gotta get rid of that
                target_ner_group_label = target_ner_group_label.replace('True', '1')
                current_sample.append((token, target_ner_group_label))
            else:
                current_sample.append((token, target_ner_group_label))
    if current_sample:
        samples.append(current_sample)

    return samples

def split_list(data: list, train_ratio: float = 0.7) -> list:
    """
    split the given list into train, validation, and tests splits
    """
    val_ratio = (1 - train_ratio) / 2

    num_items = len(data)
    train_size = int(train_ratio * num_items)
    val_size = int(val_ratio * num_items)

    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]

    return train_data, val_data, test_data

def create_hf_dataset(dataset_split: list, label_names: list, label2id: dict):
    """
    DESCRIPTION:

    Create a hf-type dataset using the 'datasets' library
    
    ARGS:
    
    dataset_split: expects the 'split', i.e. the sublist from the whole with the data
    label_names: the list with all the label name (the list comes from the whole dataset)
    label2id: label to id dictionary; it is created earlier from the label_names variable

    OUTPUT:
    
    dataset: we output here dataset object 

    """
    # Define the features of the dataset
    features = datasets.Features(
        {
            "id": datasets.Value("string"),
            "tokens": datasets.Sequence(datasets.Value("string")),
            "ner_tags": datasets.Sequence(datasets.features.ClassLabel(names=label_names)),
        }
    )

    # Create an empty dataset
    dataset = datasets.Dataset.from_dict({"id": [],"tokens": [], "ner_tags": []}, features=features)

    sample_id = 0
    # Loop over the structure and add each sublist as an example
    for sublist in dataset_split:
        # Unzip the sublist into tokens and tags
        tokens, tags = zip(*sublist)
        # Convert the tokens and tags into lists
        tokens = list(tokens)
        tags = list(tags)
        tags = [label2id[i] for i in tags]
        sample_id += 1
        # Append the example to the dataset
        dataset = dataset.add_item({"id": str(sample_id), "tokens": tokens, "ner_tags": tags})

    return dataset
