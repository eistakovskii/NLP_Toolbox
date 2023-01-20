import spacy
nlp = spacy.load("en_core_web_sm")
import cld3
from collections import Counter
import string, os, io, json, re
import nltk
nltk.download('punkt')
from nltk.util import ngrams
from nltk import word_tokenize as word_tokenize
from tqdm import tqdm
import razdel, pymorphy2
morph = pymorphy2.MorphAnalyzer() 

class vw_dataset:
  
  """
  INIT ARGS:

  texts: list of docs or list of strings
  which_n_grams: list of integers where each integer specifies which n-grams you require. Unigrams are implied
  main_dir_path: specify the path where all the scripts and script related files are located

  """
  
  def __init__(self, texts: list, which_n_grams: list, main_dir_path: str):
    self.texts = texts
    self.which_n_grams = which_n_grams
    self.main_dir_path = main_dir_path 
    self.out_list = list() # Final list with the incoming documents transformed into vowpal format

  def get_vw_dataset(self, lim_n_gr = 2, verbose=False) -> list:
    """
    DESCRIPTION:

    Takes a list of texts/documents and outputs a list of documents tokenized and counted
    in the Vowpal Wabbit format, e.g.:
      doc1 Alpha Bravo:10 Charlie:5  
      doc2 Bravo:5 Delta Echo:3
    
    INPUT:

    texts: list of docs
    lim_n_gr:  cut off limit to keep n_grams only w/ count >= lim_n_gr, default = 2
    verbose: specify verbosity

    OUTPUT:

    produces a list of strings where each element equal a document in a vowpal format

    """
    self.lim_n_gr = lim_n_gr
    self.verbose = verbose
    
    for num, text in tqdm(enumerate(self.texts)):
      
      ###### INTRO FORMATING ########
      cur_list = list()
      cur_list.append('doc'+ str(num))
      
      ######## PREPROCESSING ######
      language = self.cld3_model(text) # Detect language
      all_tokens = list(filter(lambda i: i != '-', self.clean_text(language[0]['Language'], text)))
      all_tokens_1 = " ".join(all_tokens)
      ######## GET N_GRAMS ######
      
      n_grams = nltk.word_tokenize(all_tokens_1) # get uni_grams
      dict_t_tokens = dict(Counter(n_grams))
      
      for n in self.which_n_grams:
        n_grams_curr = self.get_ngrams(all_tokens_1, n) # get n_grams
        dict_t_tokens_curr = {k:v for k, v in dict(Counter(n_grams_curr)).items() if v>=self.lim_n_gr} # Keep n_grams only w/ count >= 2
        dict_t_tokens.update(dict_t_tokens_curr) # Concatenate two dictionnaries
      
      ####### FINAL FORMATING #######
      out_cur_list = ''
      for key, value in dict_t_tokens.items():
        cur_list.append(key+':'+str(value)) 
        out_cur_list = ' '.join(cur_list)
  
      self.out_list.append(out_cur_list)

    return self.out_list

  def get_ngrams(self, text: str, n: int) -> list:
    """
    DESCRIPTION:
    
    Get n-grams
    
    INPUT:
    
    text: a string to tokenize
    n: an integer which specifies the number of tokens to include in a n-gram

    OUTPUT:
    
    returns a list of n-grams

    """
    n_grams_f = ngrams(nltk.word_tokenize(text), n)
    return ['_'.join(grams) for grams in n_grams_f]
    
  def cld3_model(self, text, num_langs=1):
    """
    Identifies language used
    """
    result_lang = []
    lang_array = [lang for lang in cld3.get_frequent_languages(text, num_langs)]
    with io.open(f"{self.main_dir_path}/Tesseract_lang_list.json", 'r', encoding='utf-8') as file:
        tsrct_lang = json.load(file)
    for item in range(0, len(lang_array)):
        language = lang_array[item][0]
        lang_code = "".join([value[0] for key, value in tsrct_lang.items() if language in key])
        lang_name = "".join([value[1] for key, value in tsrct_lang.items() if language in key])
        result_lang.append(
            {
                'Language': language,
                'Alpha3' : lang_code,
                'Language_name': lang_name,
            }
        )
    return result_lang

  def clean_text(self, lang, text):
    """
    Cleans the text from punctuation and various noise
    """
    try:
        PUNCTUATION = "–.—,€$¶№%£♫♪░▒▓♫│♪╖╣║╕╗╝┤╡╢▸╛�®‰™⏰⌚⌛⏳©¬³°¯²§±‡†¾½¼µ☒☐•!\"#$%&\\'()*+,“”/:;<=>?[\\\\]^_`{|}~’‘«»…"
        rm_multispace_text = re.sub(r'\s+', ' ', text, flags=re.I)
        rm_num_ex_text = re.sub(r'[0-9.]+(-[0-9.]+)?', '', rm_multispace_text, flags=re.I)
        rm_num_ex_lang = re.sub(r'[0-9.]+(-[0-9.]+)?', '', lang, flags=re.I)
        rm_punct_text = ''.join([ch if ch not in PUNCTUATION else '' for ch in rm_num_ex_text])
        temp_t = [_.text for _ in razdel.tokenize(rm_punct_text)]
        delete_dublicates = ' '.join([temp_t[j] for j in range(len(temp_t) - 1) if temp_t[j] != temp_t[j + 1]])
        rr = re.sub(r'[\w-]*.jpeg|[\w-]*.png|[\w-]*.jpg', '', delete_dublicates, flags=re.I)

        return self.remove_stop_words(rm_num_ex_lang, ''.join(rr).strip())

    except TypeError:
        return "Empty file"

  def remove_stop_words(self, lang, text):
    """
    Removes stop words
    """
    stop_words_array, tokens_filtered = [], []
    if lang.lower() == 'ru' or lang.lower() == 'русский' or lang.lower() == 'russian' or \
            lang.lower() == 'rus':
        with io.open(f"{self.main_dir_path}/ru.txt", 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                stop_words_array.append(line)
            file.close()
        tokens = [_.text for _ in razdel.tokenize(text.lower())]
        for token in tokens:
            if token.isdigit():

                tokens_filtered.append(token)
            elif not token in stop_words_array:
              lemma = morph.parse(token)[0].normal_form
              tokens_filtered.append(lemma)
        result = " ".join(filter(lambda x: str(x) if x is not None else '', tokens_filtered))
        tokens_filtered = [_.text for _ in razdel.tokenize(result.lower())]

    elif lang.lower() == 'en' or lang.lower() == 'english' or lang.lower() == 'eng':
        with io.open(f"{self.main_dir_path}/en.txt", 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                stop_words_array.append(line)
            file.close()
        tokens = [_.text for _ in razdel.tokenize(text.lower())]
        for token in tokens:
            if not token in stop_words_array:
              doc = nlp(token)
              for item in doc:
                lemma = item.lemma_
                tokens_filtered.append(lemma)
        result = " ".join(filter(lambda x: str(x) if x is not None else '', tokens_filtered))
        tokens_filtered = [_.text for _ in razdel.tokenize(result.lower())]
    return tokens_filtered
