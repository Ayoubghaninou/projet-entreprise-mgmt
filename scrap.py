import time
import dateparser
from dotenv import load_dotenv
import os
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException, ElementClickInterceptedException
from bs4 import BeautifulSoup
from selenium.webdriver.common.keys import Keys

# from openpyxl import Workbook
import pandas as pd

import json
import requests
import re


with open('config.json') as json_file:
    config = json.load(json_file)
load_dotenv()
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

URL = config['URL']
DriverLocation = config['DriverLocation']


def get_place_coordinates(url): 

    # Extract latitude and longitude from URL using regular expression
    match = re.search(r'@(\-?\d+\.\d+),(\-?\d+\.\d+)', url)
    if match:
        lat = match.group(1)
        lng = match.group(2)
        return lat, lng
    else:
        return None, None

def reverse_geocode(lat, lng):
    result = None
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={GOOGLE_MAPS_API_KEY}"
    response = requests.get(geocode_url)
    if response.status_code == 200:
        data = response.json()
        if len(data['results']) > 0:
            result = data['results'][0]
    return result

def render_displacy(doc):
    svg = displacy.render(doc, style='dep')
    drawing = svg2rlg(io.StringIO(svg))
    png_image = renderPM.drawToString(drawing, fmt='PNG')
    return png_image


def get_data(driver, agence):
    """
    this function get main text, score, name
    """
    print('get data...')

    lat, lng = get_place_coordinates(driver.current_url)
    result = reverse_geocode(lat, lng)
    print(result)
    city, region, country = ('','','')
    if result is not None:
        for component in result['address_components']:
            if 'locality' in component['types']:
                city = component['long_name']
            if 'administrative_area_level_1' in component['types']:
                region = component['long_name']
            if 'country' in component['types']:
                country = component['long_name']

    time.sleep(5)

    counter = count_reviews(driver)
    scrolling(driver, counter)
    more_elements = driver.find_elements_by_class_name('w8nwRe kyuRq')
    # more_elements = driver.find_elements_by_class_name('w8nwRe kyuRq')
    for list_more_element in more_elements:
        retries = 5
        for i in range(retries):
            try:
                WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CLASS_NAME, 'w8nwRe kyuRq')))
                driver.execute_script("arguments[0].scrollIntoView();", list_more_element)
                driver.execute_script("arguments[0].click();", list_more_element)
                break
            except ElementClickInterceptedException:
                print(f'Interception error on attempt {i+1}, retrying...')
                time.sleep(2)  # Wait for 2 seconds before trying again
            except Exception as e:
                print('Failed to click on the element. Error: ', str(e))

    elements = driver.find_elements_by_class_name(
        'jftiEf')
    lst_data = []
    for data in elements:
        text = ''
        try:
            wait = WebDriverWait(driver, 5)
            name_element = wait.until(EC.visibility_of_element_located((By.XPATH, './/*[contains(@class, "d4r55")]')))
            name = name_element.text
            print("name: ", name)
        except TimeoutException:
                print('Failed to retrieve some data. Moving to next review.')
                pass
            # Check if 'See more' button exists and click it if it does
        try:
            see_more_button = WebDriverWait(data, 5).until(EC.presence_of_element_located(
                (By.XPATH, './/button[contains(@class, "w8nwRe")]')))
            see_more_button.click()
        except TimeoutException:
                pass
        try:
            text_element = WebDriverWait(data, 5).until(EC.presence_of_element_located(
                (By.XPATH, './/*[contains(@class, "MyEned")]/span[1]')))
            text = text_element.text
            print("text", text)
        except TimeoutException:
            print('Failed to retrieve some data. Moving to next review.')
            pass
        try:
            date_element = WebDriverWait(data, 5).until(
                EC.presence_of_element_located((By.XPATH, './/*[contains(@class, "rsqaWe")]')))
            date = parse_relative_date(date_element.text)

            print("date", date)
        except TimeoutException:
            print('Failed to retrieve some data. Moving to next review.')
            pass
        try:
            score_element = WebDriverWait(data, 5).until(
                EC.presence_of_element_located((By.XPATH, './/*[contains(@class, "kvMYJc")]')))
            stars = score_element.find_elements_by_xpath(
                './/*[contains(@class, "vzX5Ic")]')
            score = len(stars)
            print("score", score)
        except TimeoutException:
            print('Failed to retrieve some data. Moving to next review.')
            pass

        lst_data.append([name + " from GoogleMaps", text, score, date, city, region, country])

    return lst_data


def parse_relative_date(string):
    try:
        date = dateparser.parse(string, languages=['fr'])
        if date is None:
            date = pd.Timestamp.now()  # Replace with a default date
    except Exception as e:
        print('Failed to parse date:', str(e))
        date = pd.Timestamp.now()  # Replace with a default date
    return date


def count_reviews(driver):

    button_avis = driver.find_elements(By.CLASS_NAME, 'hh2c6')[1].click()
    print(button_avis)
    time.sleep(10)
    nombre_avis = driver.find_element(By.CLASS_NAME, 'jANrlb')
    nombre_avis = int(nombre_avis.text.split("\n")[1].split(" ")[0])
    # result = driver.find_element_by_class_name(
    #     'jANrlb').find_element_by_class_name('fontBodySmall').text
    # result = result.replace(',', '')
    # result = result.split(' ')
    # result = result[0].split('\n')
    # return int(int(result[0])/10)+1
    return nombre_avis


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
    cols = ["name", "comment", 'rating', 'date', 'city', 'region', 'country']
    df = pd.DataFrame(data, columns=cols)
    
    # Check if file exists to avoid writing the header multiple times
    if not os.path.isfile('out.csv'):
        df.to_csv('out.csv', header='column_names', index=False)
    else: 
        df.to_csv('out.csv', mode='a', header=False, index=False)


def generate_urls_from_agencies():
    url = "https://www.pole-emploi.fr/annuaire/votre-pole-emploi.html"
    reponse = requests.get(url)
    page = reponse.content
    soup = BeautifulSoup(page, "html.parser")

    noms_agence = soup.find_all('a', class_='block-item-link')

    all_agences = [nom.text for nom in noms_agence]
    liste_agence = [item.replace('\n', '').replace('\xa0', '').strip() for item in all_agences]

    # urls = ["https://www.google.fr/maps/search/" + agence for agence in liste_agence]
    return liste_agence

def main_scrap():
    print('starting...')
    try:
        options = webdriver.ChromeOptions()
        options.add_argument("--lang=fr-FR")
        options.add_experimental_option('prefs', {'intl.accept_languages': 'fr,fr_FR'})
        driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
        driver.get(URL)
        liste_agence = generate_urls_from_agencies()[:10] # Assuming you've defined generate_urls_from_agencies() above.
        print(liste_agence)

        try:
            # Wait up to 10 seconds for the element to be present
            element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (By.XPATH, '//*[@id="yDmH0d"]/c-wiz/div/div/div/div[2]/div[1]/div[3]/div[1]/div[1]/form[2]/div/div/button'))
            )
            element.click()
        except:
            pass
        for agence in liste_agence:
            # Attendre jusqu'à ce que l'élément soit visible et interactif
            wait = WebDriverWait(driver, 10)
            recherche = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'tactile-searchbox-input')))

            #On clique sur la barre de recherche et on ecrit le nom d'une agence 
            recherche = driver.find_element(By.CLASS_NAME, 'tactile-searchbox-input')
            recherche.send_keys(agence)
            time.sleep(1)
            recherche.send_keys(Keys.ENTER)
            time.sleep(3)
            # Find the suggestions
            suggestions = driver.find_elements(By.CSS_SELECTOR, 'div.Nv2PK.tH5CWc.THOPZb')
            if len(suggestions) > 1:
                clickable = suggestions[0].find_element(By.XPATH, './/a')  # Adjust the locator to suit your needs
                clickable.click()
                time.sleep(3)

                data = get_data(driver, agence)
                write_to_csv(data)

                recherche.clear()
                # recherche.send_keys(agence)
                time.sleep(1)
            else:
                recherche.send_keys(Keys.ENTER)
                time.sleep(3)
                data = get_data(driver, agence)
                write_to_csv(data)

        driver.close()
        print('Done!')
        return "Success"
    except Exception as e:
        print('An error occurred:', str(e))
        return str(e)
if __name__ == "__main__":
    main_scrap()
