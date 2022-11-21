Here you can find various code for augmenting your data, mainly by scraping

* [*scr_reverso_extended.py*](https://github.com/eistakovskii/NLP_projects/blob/main/DATA_AUGMENTATION/scr_reverso_extended.py) - here I used selenium to visit a context reverso page for each word from a given list and scrape everything that appears on the screen: quick and dirty way to gather some data
![alt text](https://github.com/eistakovskii/NLP_projects/blob/main/DATA_AUGMENTATION/reverso_scr.png)

* [*scr_dict_cc.py*](https://github.com/eistakovskii/NLP_projects/blob/main/DATA_AUGMENTATION/scr_dict_cc.py) - again selenium but in this case I extract only specific words: I wrote this code to extract all the german words with tags - vulgar, rude, coll etc.
![alt text](https://github.com/eistakovskii/NLP_projects/blob/main/DATA_AUGMENTATION/dict_scr.png)

* [*scr_dict_cc.py*](https://github.com/eistakovskii/NLP_projects/blob/main/DATA_AUGMENTATION/scr_reverso_freq.py) - here I used Selenium to extract most frequent words/translations of the english toxic words and phrases of French and German from context Reverso pages
![alt text](https://github.com/eistakovskii/NLP_projects/blob/main/DATA_AUGMENTATION/scr_reverso_freq.png)
