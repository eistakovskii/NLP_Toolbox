**How to transform a CONLL type dataset for NER into a Huggingface dataset**

* First, use the open source library [label studio](https://labelstud.io/) to manually tag your data.
![alt text](https://github.com/eistakovskii/NLP_projects/blob/main/NER/CONLL_TO_HF/label_studio_ex.png)

* Export your data as a CONLL file.

* Use the [*split_data.py*](https://github.com/eistakovskii/NLP_projects/blob/main/NER/CONLL_TO_HF/split_data.py) file to split the whole tagged CONLL file into 3 txt files, 3 splits: train.txt, valid.txt, test.txt

* Transform these text files into a hugging face dataset using the [*from_conll_to_hf.py*](https://github.com/eistakovskii/NLP_projects/blob/main/NER/CONLL_TO_HF/from_conll_to_hf.py) script.

Below you will find the example code to prep your data and start training the NER model with a custom hugging face dataset.


```python

!wget https://raw.githubusercontent.com/eistakovskii/NLP_projects/main/NER/CONLL_TO_HF/from_conll_to_hf.py

from from_conll_to_hf import *

main_path_in = YOUR PATH # insert here the path (as a string) to your txt files that were split beforehand into train, valid and test splits 
tg_in = ("B-LOC", "I-LOC", "B-ORG", "I-ORG", "O") # add here your NE tags in the BIO format as a tuple
dataset = HF_NER_dataset(mp = main_path_in, tg = tg_in).dataset

print(dataset['train'])
print(dataset['test'])
print(dataset['validation'])

print("List of tags: ", dataset['train'].features['ner_tags'].feature.names)


print("First sample: ", dataset['train'][0])

```


