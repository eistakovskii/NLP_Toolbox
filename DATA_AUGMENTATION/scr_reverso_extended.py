from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from tqdm.auto import tqdm

import time

def pull_from_reverso(path_driver: str, path_curr_dir: str, path_target_words: str, your_login: str, your_password, verbose: bool = False):
    
    LOGIN_PAGE = "https://account.reverso.net/Account/Login?returnUrl=https%3A%2F%2Fcontext.reverso.net%2F&lang=en" # the loging page
    ACCOUNT = your_login # add here your login
    PASSWORD = your_password # add here your password
    
    driver = webdriver.Chrome(executable_path=path_driver)
    actions = ActionChains(driver)

    wait = WebDriverWait(driver, 30)
    
    driver.get(LOGIN_PAGE)
    wait.until(EC.element_to_be_clickable((By.NAME, "Email"))).send_keys(ACCOUNT)
    wait.until(EC.element_to_be_clickable((By.ID, "Password"))).send_keys(PASSWORD)
    wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="account"]/div[7]/button'))).click()

    to_save_text = list()

    with open(path_target_words, mode='r', encoding='utf-8') as f:
        target_words = f.readlines()
        target_words = [i.strip('\n') for i in target_words]

    for w in tqdm(target_words):             
        
        print(f'\nEXTRACTING FOR: {w}')
        driver.get(f"https://context.reverso.net/translation/russian-english/{w}") # get the page with the word

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") # scroll to the end of the page
        
        time.sleep(2) # wait until fully loaded
        skipped = 0
        
        num_words_av = int(driver.find_element(By.XPATH, f'/html/body/div[4]/section[1]/div[5]/p/span[1]').text) # number of sentences to be possibly extracted

        print(f'NUMBER OF WORDS TO EXTRACT: {num_words_av}')

        for i in range(1, num_words_av+1):
            
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") # scroll to the end of the page
            
            try:
                show_more_button = driver.find_element(By.XPATH, f'/html/body/div[4]/section[1]/div[2]/section[6]/div[2]/button[1]') # find the button loading the sentences
                actions.scroll_to_element(show_more_button).click(show_more_button).perform() # scroll to and press the button
            except:
                pass
            
            try:
                out_text = driver.find_element(By.XPATH, f'/html/body/div[4]/section[1]/div[2]/section[4]/div[{i}]/div[1]/span').text # get the i-th sentence

                to_save_text.append(out_text) 
                if not i % 100:
                    if verbose: print(f'NUMBER OF EXTRACTED SO FAR: {len(to_save_text)}')
                    time.sleep(1)
            except:
                skipped += 1
                continue
        if verbose:
            print(f'NUMBER OF SKIPPED: {skipped}')
            print(f'TORAL NUMBER OF EXTRACTED: {len(to_save_text)}')

    with open(rf'{path_curr_dir}\pulled_texts_new2_{len(to_save_text)}.txt', mode='w', encoding='utf-8') as f2:
        for i in to_save_text:
            f2.write(i+'\n')
    driver.close() 
    
    pass
