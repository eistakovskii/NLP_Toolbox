from selenium import webdriver
from selenium.webdriver.common.by import By

from tqdm.auto import tqdm

import re

driver = webdriver.Chrome(executable_path=r'YOUR_PATH')

print('\nSTARTING...')

out_list = list()

not_extracted = 0

block = ['idiom ', 'cloth.', 'hist.', 'art.', 'relig.', 'meteo.', 'hist.', 'anat.', 'cosmet.', 'gastr.', 'unit', 'ecol.', 'med.', 'sports', 'pol.', 'econ.', 'zool.','geogr.','sociol.', 'lit.', 'ling','RealEst.', 'naut.', 'journ.', 'lit.', 'sb.', 'ethn.', 'curr. neol.', 'sth.', 'publ.', 'comm. FoodInd.', 'educ.', 'orn.', 'neol.', 'fin.', 'mus.', 'spec.', 'psych.', 'mil.','comm.', 'automot.', 'FoodInd.', 'agr. equest.', 'constr.', 'archi.', 'agr.']

for i in tqdm(range(1, 4)):
    driver.get(f"https://www.dict.cc/?s=%5Boffensive%5D&pagenum={i}")
    if i == 3:
        for i in range(1,42):
            try:
                x = driver.find_element(By.XPATH, f'//*[@id="tr{i}"]/td[2]').text
                if x in block:
                    continue
                x = re.sub(r'\[[^)]*\]', '', x)
                x = re.sub(r'\([^)]*\)', '', x)
                x = re.sub(r'\<[^)]*\>', '', x)
                x = re.sub(r'\{[^)]*\}', '', x)
                x = " ".join(x.split())
                if x.startswith('to'):
                    x = x[3:]
                if x in out_list:
                    continue
                for i in block:
                    if i in x:
                        x = x.replace(i, '')
                x = " ".join(x.split())
                x = re.sub(r'\[[^)]*\]', '', x)
                out_list.append(x)
            except:
                not_extracted += 1
                continue
    else:
        for i in range(1, 51):
            try:
                x = driver.find_element(By.XPATH, f'//*[@id="tr{i}"]/td[2]').text
                x = re.sub(r'\[[^)]*\]', '', x)
                x = re.sub(r'\([^)]*\)', '', x)
                x = re.sub(r'\<[^)]*\>', '', x)
                x = re.sub(r'\{[^)]*\}', '', x)
                x = " ".join(x.split())
                if x.startswith('to'):
                    x = x[3:]
                if x in out_list:
                    continue
                for i in block:
                    if i in x:
                        x = x.replace(i, '')
                x = " ".join(x.split())
                x = re.sub(r'\[[^)]*\]', '', x) 
                out_list.append(x)
            except:
                not_extracted += 1
                continue
print('\nEXPORTING...') 

out_list = set(out_list)

with open('offen_en.txt', 'w', encoding='utf-8') as f:
    for i in out_list:
        f.write(i+'\n')
print(f'\nNUMBER OF LOST: {not_extracted}')

print('\nENDING...') 

driver.close()
