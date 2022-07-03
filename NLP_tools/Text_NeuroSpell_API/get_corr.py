import requests
import json
import time
import pandas as pd

def correct_with_ns(path: str, num_sent: int, verbose = False, verbose_checkpoint = True):
  """Accepts a text file and using API of NeuroSpell returns two csv with sentences without correction and with corrections

  Args:
      path (str): path to the text file
      num_sent (int): number of sentences that you want to check
  
  Returns:
      count_cl: "clean" sentences, count_corr: sentences with corrections, total: total time of execution excluding data wrangling, 
      num_of_dupl: number of duplicates, skipped_list: skipped sentences during the foor loop API calls
  """

  # READ THE TEXT FILE
  with open(path, 'r+') as f:
    lines = f.readlines()
    lines1 = [i.strip() for i in lines]
  
  # DATA PREPROCESSING
  df = pd.DataFrame(lines1, columns = ['Input'])
  duplicateRows = df[df.duplicated()]
  num_of_dupl = len(duplicateRows)
  df_r = df.drop_duplicates(ignore_index=True)
  lines_no_d = df_r.values.tolist()
  lnds = [lines_no_d[i][0] for i in range(len(lines_no_d))]
  test_batch = lnds[0:num_sent]

  # API PREP
  url = "http://52.59.173.227:8080/NeuroSpellRestServer/rest/NeuroSpellAll"
  headers = {'Content-Type': 'application/json'}
  payloads = [json.dumps({"language": "de", "text": lines}) for lines in test_batch]

  # INITIATE CLEAN DATAFRAMES FOR SENTENCES WITHOUT CORRECTION AND WITH
  df_clean = pd.DataFrame(columns=['Input', 'Response'])
  df_corr = pd.DataFrame(columns=['Input', 'Output', 'Response'])

  t0 = time.time() # FOR CALCULATING TIME
  
  # INITIALIZE COUNTERS
  count_cl = 0 
  count_corr = 0

  # INITIALIZE THE LIST OF SKIPPED SENTENCES
  skipped_list = []  

  # MAKE API CALLS AND COUNT CLEAN AND CORRECTED
  for ind, pls in enumerate(payloads):
    try:
      response = requests.request("POST", url, headers=headers, data=pls) # DO THE API CALLS
      temp = json.loads(response.text)
      correction = temp['corrected']
      if verbose == True:
        print('Input: ', test_batch[ind])
        print('Output: ', correction)
      if test_batch[ind] == correction: # CATCH NO CORRECTIONS
        if verbose == True:
          print('No correction')
        count_cl += 1
        df_clean.loc[count_cl] = [test_batch[ind], response.text]
      else: # CATCH CORRECTIONS
        if verbose == True:
          print('Sentence was corrected')
        count_corr += 1
        df_corr.loc[count_corr] = [test_batch[ind], correction, response.text]
      if ind % 1000 == 0: # CREATE CHECKPOINTS
        df_checkpoint_c = df.clean
        df_checkpoint_corr = df_corr
        df_checkpoint_corr.to_csv('corrected'+ str(ind) + '.csv', encoding='utf-8')
        df_checkpoint_c.to_csv('clean' + str(ind) + '.csv', encoding='utf-8')
        if verbose_checkpoint == True: # REPORT THE CHECKPOINT
          print(str(ind) + ' sentences out of '+ str(num_sent) + ' done!\n')
          t2 = time.time()
          total2 = t2-t0
          print('Execution time:', total2)
          print('\n')
      elif ind % 500 == 0:
        if verbose_checkpoint == True: # REPORT THE SMALLER CHECKPOINT
          print(str(ind) + ' sentences out of '+ str(num_sent) + ' in progress!')
          print('NO CSV EXPORTED\n') 
    except: # REPORT SKIPPED SENTENCES IF ANY
      if verbose == True:
        print('Failed to retrieve\n')
      skipped_list.append(test_batch[ind])
      continue
  
  t1 = time.time() # FOR CALCULATING TIME

  total = t1-t0 # FOR CALCULATING TIME
  
  # OUTPUT FINAL CSVs WITH THE THE COLLECTED CLEAN AND CORRECTED SENTENCES
  df_corr.to_csv('corrected.csv', encoding='utf-8')
  df_clean.to_csv('clean.csv', encoding='utf-8')

  if verbose == True:
    print('\nExecution time: ', total)

    print('\nNumber of corrected out of {}: {}'.format(num_sent, count_corr)) 
  
  return count_cl, count_corr, total, num_of_dupl, skipped_list