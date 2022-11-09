**How to transform a CONLL type dataset for NER into a Huggingface dataset**

* First, use the open source library [label studio](https://labelstud.io/) to manually tag your data.
![alt text](https://github.com/eistakovskii/NLP_projects/blob/main/NER/CONLL_TO_HF/label_studio_ex.png)

* Export your data as a CONLL file.

* Import and convert the CONLL file into a hugging face dataset using the [*from_conll_to_hf.py*](https://github.com/eistakovskii/NLP_projects/blob/main/NER/CONLL_TO_HF/from_conll_to_hf.py) script.

Note that the *from_conll_to_hf.py* script does the split into train, validation, and test splits for you internally. The proportion is 80/10/10 respectively.
Change the (*split_data.py*) if you prefer a different split

Below you will find the example code to prep your custom hugging face dataset inside a jupyter notebook.

  Clone the repor
  ``` shell
  git clone https://github.com/eistakovskii/NLP_projects/CONLL_TO_HF.git
  ```
  Locate to the repo directory
  
  Run the commands below
  ``` shell
  pip install requirements.txt
  py ner_create_dataset.py --file_path PATH_TO_YOUR_FILE\YOUR_FILE.conll --tags LOC,ORG,MISC --export True

  ```
Below you will find the example code to prep your custom hugging face dataset inside a jupyter notebook

  ```
  !wget https://raw.githubusercontent.com/eistakovskii/NLP_projects/main/NER/CONLL_TO_HF/from_conll_to_hf.py
  !wget https://raw.githubusercontent.com/eistakovskii/NLP_projects/main/NER/CONLL_TO_HF/split_data.py
  ```

  ```python
  from from_conll_to_hf import *

  main_path_in = YOUR PATH # insert here the path (as a string) to your txt file
  tg_in =  ['ORG', 'LOC', 'MISC'] # specify here your target NE tags. Note that they will be converted into the BIO format and the tag 'O' will be added by default
  dataset = HF_NER_dataset(mp = main_path_in, tg = tg_in).dataset

  print(dataset['train'])
  print(dataset['test'])
  print(dataset['validation'])

  print("List of tags: ", dataset['train'].features['ner_tags'].feature.names)


  print("First sample: ", dataset['train'][0])

  ```


