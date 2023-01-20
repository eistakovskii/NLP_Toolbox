**Text to Vowpal Wabbit dataset**

[The vowpal_dataset class](https://github.com/eistakovskii/NLP_projects/blob/main/TOPIC_MODELLING/TEXT_TO_VOWPAL_WABBIT/vowpal_wabbit_dataset.py) helps you to turn your incoming texts in Russian (inside a list) into a list with documents converted from plain text to [*Vowpal Wabbit*](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Input-format) format which later can be freely used for the topic modelling task.
Originally the class was written to be used for topic modelling with [BigARTM](https://github.com/bigartm/bigartm).

**How to transform your txt file into Vowpal Wabbit dataset**

* Make sure that you created a txt file with each document/text placed on a separate line

* Clone the repo

``` shell
git clone https://github.com/eistakovskii/NLP_projects/TOPIC_MODELLING/TEXT_TO_VOWPAL_WABBIT.git 
```
* Run the commands below
``` shell
pip install requirements.txt
py vowpal_wabbit_dataset.py --text_file_path PATH_TO_YOUR_FILE\YOUR_FILE.text --which_n_grams 2,3 --dir_path PATH_TO_YOUR_DIR
```
