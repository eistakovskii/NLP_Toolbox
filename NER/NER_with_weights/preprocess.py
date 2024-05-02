import json
import os
from random import shuffle
import numpy as np
from datasets import DatasetDict
import datasets
from collections import Counter
from argparse import ArgumentParser
import yaml


from preprocessing_utils import do_the_preprocessing, split_list, create_hf_dataset
from utils import set_random_seed, TARGET_NER_GROUPS

# TODO: limit the amount of data used for testing, e.g. instead of 50k sample only 5k or 5% of all the data
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--path_to_raw_data',
        required=True,
        type=str, 
        help="Path to the data txt file, e.g. '\data\raw\augmentation_output_new.txt'"
    )
    parser.add_argument(
        '--to_save_path',
        required=True,
        type=str, 
        help="Path to save files after preprocessing, e.g. '\data\processed'"
    )

    args = parser.parse_args()

    curr_dir = os.getcwd()

    data_path = os.path.join(curr_dir, args.path_to_raw_data) 
    output_path = os.path.join(curr_dir, args.to_save_path)
    os.makedirs(output_path, exist_ok=True)

    params = yaml.safe_load(open("params.yaml"))[f"preprocessing_stage_{curr_domain}"]

    seed_number = params["seed"]
    split_ratio = float(params["split_ratio"]) # cast to float since apparently yml poorly works with floats...

    set_random_seed(seed_number) # set the random seed

    main_dataset = do_the_preprocessing(data_path, NER_GROUP) # load from conll format (.txt) into list of lists
    total_samples_number_present = len(main_dataset)

    shuffle(main_dataset) # shuffle NB: we fixed the seed

    train_split, val_split, test_split = split_list(main_dataset, split_ratio) # split into the splits NB: the ratio can be adjusted via arguments
    
    total_samples_per_split = {'train': len(train_split), 'validation': len(val_split), 'test': len(test_split)}

    # if NER_GROUP == 'entity': # Here starts the part where we deal with the unbalanced dataset and create label weights; 
    #     # note that we apply this logic only to the 'entity' ner group since other groups did not benefit from loss weights.
    #     # I attribute that to the fact that in the other groups we have simpler NER labels, e.g. argument vs O or 1/2 vs O  
    all_labels_temp = list()
    all_labels_raw = list()

    for i in main_dataset: # here we extract the labels by iterating throught the data
        for j in i:
            all_labels_raw.append(j[1])
            if j[1] != 'O':
                all_labels_temp.append(j[1][2:])

    all_labels_counter = Counter(all_labels_raw)

    sorted_dict = dict(sorted(all_labels_counter.items(), key=lambda x: x[1], reverse=True)) # sort the items in the Counter by their values in descending order

    l_1 = list(sorted_dict.keys())

    all_labels_temp = sorted(list(set(all_labels_temp)))

    label_names = list()

    for tag in all_labels_temp:
        bio_tags_out = [prefix + tag for prefix in ['B-', 'I-']]
        label_names.extend(bio_tags_out)

    label_names.append('O')

    missing_tags = set(label_names).difference(set(l_1)) # we need that to find which tags should exist but were not present in the dataset, e.g. some "I-" tags

    artif_freqs = dict()

    for i in list(missing_tags): # artificially get the frequencies for the missing tags, e.g. if I- missing we get the frequency from its head, i.e. B- tag
        if i.startswith('I-'):
            curr_tag = i[2:]
            retr_fr = 'B-' + curr_tag
            artif_freqs[i] = sorted_dict[retr_fr]
        elif i.startswith('B-'):
            curr_tag = i[2:]
            retr_fr = 'I-' + curr_tag
            artif_freqs[i] = sorted_dict[retr_fr]

    sorted_dict.update(artif_freqs)

    all_labels_final_freqs = dict(sorted(sorted_dict.items(), key=lambda x: x[1], reverse=True)) # resort after added artificial frequencies

    all_labels_final_sorted_keys = list(all_labels_final_freqs.keys())
    # all_labels_final_sorted_values = list(all_labels_final_freqs.values())


    smooth_factor = 1e-5 # add a small constant to avoid log(0), i.e. smoothing
    smoothed_frequencies = {cls: freq + smooth_factor for cls, freq in all_labels_final_freqs.items()}

    total_samples = sum(smoothed_frequencies.values()) # total number of samples

    log_weights = {cls: np.log((total_samples / freq)) for cls, freq in smoothed_frequencies.items()} # compute logarithmic class weights with smoothing

    max_weight = max(log_weights.values())
    normalized_label_weights = {cls: weight / max_weight for cls, weight in log_weights.items()} # normalize weights

    label_names = all_labels_final_sorted_keys[:] # make a hard copy
    # else: # we execute simpler logic for all other NER groups 
    #     all_labels_temp = list()

    #     all_labels_raw = list()

    #     for i in main_dataset:
    #         for j in i:
    #             all_labels_raw.append(j[1])
    #             if j[1] != 'O':
    #                 all_labels_temp.append(j[1][2:])

    #     all_labels_temp = sorted(list(set(all_labels_temp)))

    #     label_names = list()

    #     for tag in all_labels_temp:
    #         bio_tags_out = [prefix + tag for prefix in ['B-', 'I-']]
    #         label_names.extend(bio_tags_out)

    #     label_names.append('O')

    label2id = {val:ind for ind, val in enumerate(label_names)} # from label to id (the number values of the tag in the list)
    id2label = {ind:val for ind, val in enumerate(label_names)} # vice versa


    train = create_hf_dataset(train_split, label_names, label2id) # create each split
    validation = create_hf_dataset(val_split, label_names, label2id)
    test =  create_hf_dataset(test_split, label_names, label2id)

    # passing some extra info that could be needed, i.ee weights and labels
    # if NER_GROUP == 'entity':
    # features = datasets.Features( # TODO: do we need these 'features' for the 'info' at all?????
    #     {
    #         # "weights": datasets.Sequence(datasets.Value("float64")),
    #         "labels": datasets.Sequence(datasets.Value("string")),
    #     }
    # ) 
    general_info = datasets.Dataset.from_dict({"weights": list(normalized_label_weights.values()),"labels": label_names}) # TODO: apply 'stronger' weights for other NER groups for the 'O' class
    # else: # in the case of other NER groupds we do not need the weights
    #     # features = datasets.Features({"labels": datasets.Sequence(datasets.Value("string"))}) # TODO: do we need these 'features' for the 'info' at all?????
    #     general_info = datasets.Dataset.from_dict({"labels": label_names}) # TODO: in theory i do not need to pass labels here since I have that info on the train splits
    
    main_dataset = DatasetDict({
        'train': train,
        'validation': validation,
        'test': test,
        'info': general_info
    })

    # we need this list to get the frequencies of the tags and then export it as json for a quick overview of the dataset 
    all_labels_raw = [i for i in all_labels_raw if not i.startswith('I-')] # get rid of the I-tags
    all_labels_raw = [i if not i.startswith('B-') else i[2:] for i in all_labels_raw]
    general_frequency_statistics = dict(sorted(dict(Counter(all_labels_raw)).items(), key=lambda x: x[1], reverse=True)) # cast to Counter and then sort descending

    os.makedirs(output_path, exist_ok=True) # TODO: seems to be absolete since it's already done during .save_to_disk -> check

    main_dataset.save_to_disk(os.path.join(output_path, f'{NER_GROUP}.hf')) # export the dataset to disk

    stats_json = {'ner_group': NER_GROUP, 'total_samples_number': total_samples_number_present, 'samples_number_per_split': total_samples_per_split, 'tag_frequency': general_frequency_statistics, 'seed_number': seed_number}

    with open(os.path.join(output_path, f'{NER_GROUP}.hf', 'dataset_info.json'), 'w', encoding='utf-8') as f: # output the stats_json dict as a json
        json.dump(stats_json, f, ensure_ascii=False, indent=4)
