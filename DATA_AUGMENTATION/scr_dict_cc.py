from selenium import webdriver
from selenium.webdriver.common.by import By

driver = webdriver.Chrome(executable_path=r'PATH TO YOUR DRIVER\chromedriver.exe') # ADD HERE THE PATH TO YOUR DRIVER


print('\nSTARTING...')

# x = driver.find_element(By.XPATH, '//*[@id="tr1"]').text
out_list = list()
for i in range(1, 4):
    driver.get(f"https://www.dict.cc/?s=%5Boffensive%5D&pagenum={i}")
    if i == 3:
        for i in range(1, 47):
            x = driver.find_element(By.XPATH, f'//*[@id="tr{i}"]/td[3]').text
            out_list.append(x)
    else:
        for i in range(1, 51):
            x = driver.find_element(By.XPATH, f'//*[@id="tr{i}"]/td[3]').text
            out_list.append(x)

with open('offen.txt', 'w', encoding='utf-8') as f:
    for i in out_list:
        f.write(i+'\n')

print('\nENDING...') 


driver.close()
