from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from tqdm import tqdm


def pull_from_reverso(path_driver, path_target_words, your_login, your_password, verbose=False):
    
    LOGIN_PAGE = "https://account.reverso.net/Account/Login?returnUrl=https%3A%2F%2Fcontext.reverso.net%2F&lang=en"
    ACCOUNT = your_login # add here your login
    PASSWORD = your_password # add here your password

    driver = webdriver.Chrome(executable_path=path_driver)

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
        driver.get(f"https://context.reverso.net/translation/russian-english/{w}") 

        try:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            element = driver.find_element(By.XPATH, '//*[@id="load-more-examples"]')   
            ActionChains(driver).click(element).perform()
        except:
            pass
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        try:
            for i in range(1, 100):
                try:
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    out_t = driver.find_element(By.XPATH, f'/html/body/div[3]/section[1]/div[2]/section[4]/div[{i}]/div[1]').text
                    if len(str(out_t)) != 0:
                        to_save_text.append(str(out_t))
                    else:
                        continue
                except:
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    out_t = driver.find_element(By.XPATH, f'/html/body/div[4]/section[1]/div[2]/section[4]/div[{i}]/div[1]').text
                    if len(str(out_t)) != 0:
                        to_save_text.append(str(out_t)) 
        except:
            continue

    with open(f'pulled_texts{len(to_save_text)}_norm.txt', mode='w', encoding='utf-8') as f2:
        for i in to_save_text:
            f2.write(i+'\n')



    if verbose: print('\n')
    driver.close() 
    
    pass
