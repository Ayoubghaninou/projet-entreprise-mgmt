import time
import dateparser
import os

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException

# from openpyxl import Workbook
import pandas as pd

import json

with open('config.json') as json_file:
    config = json.load(json_file)

URL = config['URL']
DriverLocation = config['DriverLocation']



def get_data(driver):
    """
    this function get main text, score, name
    """
    print('get data...')
    more_elemets = driver.find_elements_by_class_name('w8nwRe kyuRq')
    for list_more_element in more_elemets:
        list_more_element.click()

    elements = driver.find_elements_by_class_name(
        'jftiEf')
    lst_data = []
    for data in elements:
        text = ''
        try:
            name = data.find_element_by_xpath(
                './/*[contains(@class, "d4r55")]').text
            print("name: ", name)
        except TimeoutException:
                print('Failed to retrieve some data. Moving to next review.')
                pass
            # text_container = WebDriverWait(data, 3).until(EC.presence_of_element_located((By.XPATH, './/*[contains(@class, "MyEned")]')))

            # Check if 'See more' button exists and click it if it does
        try:
            see_more_button = WebDriverWait(data, 2).until(EC.presence_of_element_located(
                (By.XPATH, './/button[contains(@class, "w8nwRe")]')))
            see_more_button.click()
        except TimeoutException:
                pass
        try:        
            text_element = WebDriverWait(data, 10).until(EC.presence_of_element_located(
                (By.XPATH, './/*[contains(@class, "MyEned")]/span[1]')))
            text = text_element.text
            print("text", text)
        except TimeoutException:
            print('Failed to retrieve some data. Moving to next review.')
            pass
        try:
            date_element = WebDriverWait(data, 10).until(
                EC.presence_of_element_located((By.XPATH, './/*[contains(@class, "rsqaWe")]')))
            date = parse_relative_date(date_element.text)

            print("date", date)
        except TimeoutException:
            print('Failed to retrieve some data. Moving to next review.')
            pass
        try:
            score_element = WebDriverWait(data, 10).until(
                EC.presence_of_element_located((By.XPATH, './/*[contains(@class, "kvMYJc")]')))
            stars = score_element.find_elements_by_xpath(
                './/*[contains(@class, "vzX5Ic")]')
            score = len(stars)
            print("score", score)
        except TimeoutException:
            print('Failed to retrieve some data. Moving to next review.')
            pass

        lst_data.append([name + " from GoogleMaps", text, score, date])

    return lst_data


def parse_relative_date(string):
    date = dateparser.parse(string, languages=['fr'])
    return date

def count_reviews(driver):
    try:
        # Wait up to 10 seconds for the element to be present
        element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.XPATH, '//*[@id="yDmH0d"]/c-wiz/div/div/div/div[2]/div[1]/div[3]/div[1]/div[1]/form[2]/div/div/button'))
        )
        element.click()
    except:
        pass
    result = driver.find_element_by_class_name(
        'jANrlb').find_element_by_class_name('fontBodySmall').text
    result = result.replace(',', '')
    result = result.split(' ')
    result = result[0].split('\n')
    return int(int(result[0])/10)+1


def scrolling(driver, counter):
    print('scrolling...')
    scrollable_div = driver.find_element_by_xpath(
        '//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]')
    for _i in range(counter):
        scrolling = driver.execute_script(
            'document.getElementsByClassName("dS8AEf")[0].scrollTop = document.getElementsByClassName("dS8AEf")[0].scrollHeight',
            scrollable_div
        )
        time.sleep(3)


def write_to_csv(data):
    print('write to excel...')
    cols = ["name", "comment", 'rating', 'date']
    df = pd.DataFrame(data, columns=cols)
    
    # Check if file exists to avoid writing the header multiple times
    if not os.path.isfile('out.csv'):
        df.to_csv('out.csv', header='column_names', index=False)
    else: 
        df.to_csv('out.csv', mode='a', header=False, index=False)



def main_scrap():
    print('starting...')
    try: 
        options = webdriver.ChromeOptions()
        # options.add_argument("--headless")  # show browser or not
        options.add_argument("--lang=en-US")
        options.add_experimental_option(
            'prefs', {'intl.accept_languages': 'en,en_US'})
        DriverPath = DriverLocation
        driver = webdriver.Chrome(DriverPath, options=options)

        driver.get(URL)
        time.sleep(5)

        counter = count_reviews(driver)
        scrolling(driver, counter)

        data = get_data(driver)
        driver.close()

        write_to_csv(data)
        print('Done!')
        return "Success"
    except Exception as e:
        print('An error occurred:', str(e))
        return str(e)
if __name__ == "__main__":
    main_scrap()
