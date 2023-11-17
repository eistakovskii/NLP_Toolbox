def return_n_samples(how_many: int, dataset):
    
    """
    This function extracts N samples from the CoLa like datasets:
    Tested on RuCoLa and EnCola
    """
    
    count_0 = []
    count_1 = []
    
    count = 0

    while len(count_0) < how_many or len(count_1) < how_many:
        if count < len(dataset['train']):
            label = dataset['train'][count]['label']
            sentence = dataset['train'][count]['sentence']
            if label == 1 and len(count_1) < how_many:
                count_1.append(sentence)
            elif label == 0 and len(count_0) < how_many:
                count_0.append(sentence)
            count += 1
        else:
            break

    assert len(count_0) == len(count_1)
    
    true_labels = [0]*how_many
    true_labels.extend([1]*how_many)
    
    count_0.extend(count_1)
    
    return count_0, true_labels
