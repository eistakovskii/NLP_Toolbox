from datasets import load_dataset
from tqdm import tqdm
from random import randint, shuffle
import pandas as pd
import string 
from corruption import Corrupter

def rand_ind(num_of_idx: int, u_range: int, verbose = False) -> list:

  out_list = list()

  if verbose: print(f'\nSTARTING...\n')

  for i in tqdm(range(num_of_idx)):
      flag = 0
      n_ind = randint(0, u_range-1)
      if n_ind not in out_list:
          out_list.append(n_ind) 
      else:
          flag = 1
          while flag:
              n_ind_2 = randint(0, u_range-1)
              if n_ind_2 not in out_list:
                  out_list.append(n_ind_2) 
                  flag = 0
              else:
                  continue

  if verbose: print(f'\nFINISHED!\n')

  return out_list

def retrieve_data(ind_list: list, u_data, lang: str, verbose = False) -> list:

  if verbose: print(f'\nSTARTING...\n')

  out_list_data = list()

  for ind in tqdm(ind_list):
      out_list_data.append(u_data['train'][ind]['translation'][lang])

  if verbose: print(f'\nFINISHED!\n')

  return out_list_data

def shuffle_str(inp_str: str) -> str:

  inp_str = inp_str.split()
  shuffle(inp_str)
  inp_str = " ".join(inp_str)

  return inp_str

########################################################MAIN FUNCTION####################################################################

def get_dataset(target_lang: str, corpus_size: int, verbose = True):

  if verbose: print('\nSTARTING!!!\n')
  ################################ LOAD AND PREPROCESS DATA FROM THE DATASET ################################

  len_pairs_list = ['ar-cs', 'ar-de', 'cs-de', 'ar-en', 'cs-en', 'de-en', 'ar-es', 'cs-es', 'de-es', 'en-es',
                    'ar-fr', 'cs-fr', 'de-fr', 'en-fr', 'es-fr', 'ar-it', 'cs-it', 'de-it', 'en-it', 'es-it',
                    'fr-it', 'ar-ja', 'cs-ja', 'de-ja', 'en-ja', 'es-ja', 'fr-ja', 'ar-nl', 'cs-nl', 'de-nl',
                    'en-nl', 'es-nl', 'fr-nl', 'it-nl', 'ar-pt', 'cs-pt', 'de-pt', 'en-pt', 'es-pt', 'fr-pt',
                    'it-pt', 'nl-pt', 'ar-ru', 'cs-ru', 'de-ru', 'en-ru', 'es-ru', 'fr-ru', 'it-ru', 'ja-ru',
                    'nl-ru', 'pt-ru', 'ar-zh', 'cs-zh', 'de-zh', 'en-zh', 'es-zh', 'fr-zh', 'it-zh', 'ja-zh',
                    'nl-zh', 'pt-zh', 'ru-zh']

  target_lang_pair = None

  for i in len_pairs_list:
      if target_lang in i:
          target_lang_pair = i

  dataset_target = load_dataset("news_commentary", target_lang_pair)

  ind_b = str(dataset_target['train']).index('num_rows: ') + 10
  ind_e = str(dataset_target['train']).index('\n})')
  num_rows_total  = int(str(dataset_target['train'])[ind_b:ind_e])

  if verbose: print('\nGETTING INDICES...\n')
  inds_target = rand_ind(corpus_size, num_rows_total)

  if verbose: print('\nRETRIEVING DATA...\n')
  data_target  = retrieve_data(inds_target, dataset_target, target_lang)

  ################################ CORRUPTER STAGE ################################

  corrupter = Corrupter() # initialize corrupter


  list_ones = data_target[:int(corpus_size/2)]
  list_zeros = data_target[int(corpus_size/2):]

  if verbose: print('\nCORRUPTING DATA...\n')

  for i, v in enumerate(tqdm(list_ones)): # corrupt 50% of the data in "list_ones"
      rand_num = randint(10, 1000)
      rand_num2 = randint(0, 100)
      corr_sent = corrupter.corrupt(v, syn=0, typo=rand_num).split()
      corr_sent = ' '.join([corr_sent[i-1]+v if v in string.punctuation else v for i, v in enumerate(corr_sent)])

      if rand_num2 >= 50:
          corr_sent = shuffle_str(corr_sent)
      list_ones[i] = corr_sent


  ################################ OUTPUT CSV ################################

  if verbose: print('\nOUTPUTING CSV...\n')

  list_ones.extend(list_zeros)

  assert len(list_ones) == corpus_size, f"Mismatch of the intended corpus size: {corpus_size} and the recieved one: {len(list_ones)}"

  len_ones = len(list_ones)

  l1 = [1 for i in range(int(corpus_size/2))]
  l0 = [0 for i in range(int(corpus_size/2))]


  l1.extend(l0)


  df_out = pd.DataFrame(list(zip(l1, list_ones)), columns = ['label', 'text'])
  df_out.to_csv(f'training_data_fluency_{target_lang}_{round(len(list_ones)/1000)}k.csv', index=False, encoding='utf-8')

  if verbose: print('\nFINISHED!!!\n')

  pass
