from selenium import webdriver
from selenium.webdriver.common.by import By

from tqdm.auto import tqdm

driver = webdriver.Chrome(executable_path=r'PATH_TO_YOUR_DRIVER\chromedriver.exe')

not_extracted_fr = list()
not_extracted_de = list()
out_fr = list()
out_de = list()

print('\nSTARTING...')

with open(r'PATH_TO_YOUR_LIST_OF_TOXIC_WORDS.txt', mode = 'r', encoding = 'utf-8') as f:
    temp_data = f.readlines()
    vulg_all_data = [i.strip() for i in temp_data]

for i in tqdm(vulg_all_data):
    
    block = ['m', 'f', 'mf', 'n', 'Show more']
    
    ### FRENCH ###
    try:
        driver.get(f"https://context.reverso.net/translation/english-french/{i}")
        temp_fr = driver.find_element(By.XPATH, f'//*[@id="translations-content"]').text 
        temp_fr = temp_fr.split('\n')
        out_fr_t = [i for i in temp_fr if i not in block]
        out_fr.extend(out_fr_t)
    except:
        not_extracted_fr.append(i)

    
    ### GERMAN ###
    try:
        driver.get(f"https://context.reverso.net/translation/english-german/{i}")
        temp_de = driver.find_element(By.XPATH, f'//*[@id="translations-content"]').text 
        temp_de = temp_de.split('\n')
        out_de_t = [i for i in temp_de if i not in block]
        out_de.extend(out_de_t)
    except:
        not_extracted_de.append(i)

print(f'\nNOT EXTRACTED FRENCH: {len(not_extracted_fr)}')
print(f'\nNOT EXTRACTED GERMAN: {len(not_extracted_de)}\n')

print(f'EXPORTING...')

with open(f'not_extracted_fr_{len(not_extracted_fr)}.txt', mode='w', encoding='utf-8') as f:
    for i in not_extracted_fr:
        f.write(i+'\n')

with open(f'not_extracted_de_{len(not_extracted_de)}.txt', mode='w', encoding='utf-8') as f:
    for i in not_extracted_de:
        f.write(i+'\n')

with open(f'extracted_fr_{len(out_fr)}.txt', mode='w', encoding='utf-8') as f:
    for i in out_fr:
        f.write(i+'\n')

with open(f'extracted_de_{len(out_de)}.txt', mode='w', encoding='utf-8') as f:
    for i in out_de:
        f.write(i+'\n')

print(f'\nFINISHED!\n')

driver.close()
