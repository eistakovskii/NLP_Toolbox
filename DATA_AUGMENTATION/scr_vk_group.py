from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

import time

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

LOGIN_PAGE = "https://vk.com/"
ACCOUNT = "YOUR_LOGIN"
PASSWORD = "YOUR_PASSWORD"

driver.get(LOGIN_PAGE)

wait_sh = WebDriverWait(driver, 2)

wait = WebDriverWait(driver, 5)

wait10 = WebDriverWait(driver, 10)

wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "VkIdForm__input"))).send_keys(ACCOUNT)
wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="index_login"]/div[1]/form/button[1]/span/span'))).click()
wait10.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="root"]/div/div/div/div/div[2]/div/div/div/form/div[1]/div[3]/div[1]/div/input'))).send_keys(PASSWORD)
wait10.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="root"]/div/div/div/div/div[2]/div/div/div/form/div[2]/button[1]'))).click()
time.sleep(5) 

driver.get(f"LINK_TO_THE_GROUP_PAGE")

counter = 1
actions = ActionChains(driver)

all_texts = list()

while True:
    try:
        # print('\n')
        out_t = driver.find_element(By.XPATH, f'/html/body/div[11]/div/div/div[2]/div[2]/div[2]/div/div[1]/div[2]/div[2]/div/div[8]/div[2]/div[{counter}]/div/div[2]/div/div[1]/div/div[1]')
        if out_t.text.endswith('re'):
            show_more_button = driver.find_element(By.XPATH, f'/html/body/div[11]/div/div/div[2]/div[2]/div[2]/div/div[1]/div[2]/div[2]/div/div[8]/div[2]/div[{counter}]/div/div[2]/div/div[1]/div/div[1]/button')
            actions.scroll_to_element(out_t).click(show_more_button).perform()
            out_text = driver.find_element(By.XPATH, f'/html/body/div[11]/div/div/div[2]/div[2]/div[2]/div/div[1]/div[2]/div[2]/div/div[8]/div[2]/div[{counter}]/div/div[2]/div/div[1]/div/div[1]').text
            # print(f'\n Post №{counter} Text: {out_text}')
            all_texts.append(repr(out_text))
            counter += 1
            print(f'\nPOST № {counter}')
        else:
            # print(f'\n Post №{counter} Text: {out_t.text}')
            all_texts.append(repr(out_t.text))
            counter += 1
            print(f'\nPOST № {counter}')
        if not counter % 5:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        if not counter % 100:
            with open(f'temp2_FACTORY_CATS_{len(all_texts)}.txt', mode='w', encoding='utf-8') as f:
                for i in all_texts:
                    f.write(i+'\n')
    except:
        continue
driver.close() 
