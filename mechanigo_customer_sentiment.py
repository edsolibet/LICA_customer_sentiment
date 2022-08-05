# -*- coding: utf-8 -*-

import sys
import subprocess
import pkg_resources

required = {'pandas', 'numpy', 'selenium', 'datetime', 'seaborn', 'streamlit-aggrid'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)

# important libraries
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import cm
from adjustText import adjust_text
import matplotlib.pyplot as plt
import re, sys, string, time
from datetime import datetime

# scraping and app
import streamlit as st
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from st_aggrid import GridOptionsBuilder, AgGrid
from parsel import Selector

# import natural language processing module and tools
import nltk
from nltk.corpus.reader.reviews import Review
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
st.set_page_config(page_icon=":chart_with_upwards_trend:", page_title="MechaniGo Customer Sentiment")

# to run selenium in headless mode (no user interface/does not open browser)
options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument("--disable-gpu")
options.add_argument("--disable-features=NetworkService")
options.add_argument("--window-size=1920x1080")
options.add_argument("--disable-features=VizDisplayCompositor")

# ignore warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# get file directory
import os 
dir_path = os.path.dirname(os.path.realpath("Mechanigo Google Review Scraper_jupyter.ipynb"))

# variable to determine if need to read from file
run_cond = False

def extract_text(s):
    '''
    Cleanup review texts
    '''
    full_text = str(s).split('review-full-text" tabindex="-1" style="display:none">')[-1]
    no_span = full_text.split('</span>')[0]
    after_span = no_span.split('<span jscontroller="MZnM8e" jsaction="rcuQ6b:npT2md">')[-1]
    before_div = after_span.split('<div')[0]
    while "<br>" in before_div:
        before_div = "".join(before_div.split("<br"))
    raw = before_div.split('(Translated by Google) ')[-1]
    raw = raw.split("<hr>")[0]
    #text = cleanup(raw)
    return raw

def get_google_reviews(driver, names):

    def get_elems(response):
        a = [elem.get() for elem in response.xpath("//tbody/tr[@class]") if 'MechaniGO.ph' in elem.get()]
        return [i.split('</td>')[0] for i in a[0].split('<td class=" ">')][1:]

    google_reviews = []
    authors = []
    for name in names:
        driver.get('https://pleper.com/index.php?do=tools&sdo=analyze_google_reviewer')
        driver.find_element(By.NAME, 'url').send_keys(name[0])
        WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, '//input[@value="Analyze Google Reviews"]'))).click()
        try:
            WebDriverWait(driver, 5).until(EC.presence_of_all_elements_located((By.XPATH, '//tbody[@role="alert"]')))
            google_reviews.append((get_elems(Selector(driver.page_source))))
            authors.append(name[1])
            #print ('name: {}'.format(name[1]))
        except:
            # Can cause TimeoutException when some names have empty google review data
            continue
    a = pd.DataFrame.from_records(google_reviews, columns=["country", "city", "business_type", "raw", "date", "rating"])
    a.loc[:, 'author'] = authors
    a.loc[:, 'date'] = pd.to_datetime(a.loc[:,'date']).dt.date
    a.loc[:, 'raw'] = a['raw'].apply(lambda x: extract_text(x))
    a.loc[:, 'platform'] = ['google']*len(authors)
    return a[['date', 'author', 'raw', 'platform']]

def extract_google_reviews(driver, query):
    driver.get('https://google.com/?hl=en')
    driver.find_element_by_name('q').send_keys(query)
    WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.NAME, 'btnK'))).click()

    '''
    Locate google review link and extract total number of reviews
    Click link
    '''
    reviews_header = driver.find_element_by_css_selector('div.kp-header')
    reviews_link = reviews_header.find_element_by_partial_link_text('Google reviews')
    number_of_reviews = int(reviews_link.text.split()[0])
    reviews_link.click()

    '''
    Load all reviews and get page source
    '''
    all_reviews = WebDriverWait(driver, 3).until(EC.presence_of_all_elements_located((By.XPATH, '//span[@jscontroller="MZnM8e"]')))
    while len(all_reviews) < number_of_reviews:
            driver.execute_script('arguments[0].scrollIntoView(true);', all_reviews[-1])
            all_reviews = driver.find_elements(By.XPATH, '//span[@jscontroller="MZnM8e"]')
    page_content = driver.page_source
    response = Selector(page_content)

    names = [[name.get().split('"')[1].split("?hl")[0], name.get().split(">")[1].split("<")[0]]
             for name in response.xpath('//div[@class="TSUbDb"]/a[contains(@href, "https://www.google.com/maps/contrib/")]')]
    
    return get_google_reviews(driver, names)

def extract_fb_reviews(csv_url):
    df_fb = pd.read_csv(csv_url)
    fb_reviews = [extract_text(rev) for rev in df_fb.loc[:,'message']]
    return pd.DataFrame({'date': pd.to_datetime(df_fb.loc[:,'postDate'].apply(lambda x: x[:10] if type(x)==str else 0)).dt.date, 'author': df_fb.loc[:,'fullName'], 
                         'raw' : fb_reviews, 'platform': ['facebook']*len(fb_reviews)})
        

if __name__ == '__main__':
    try:
        #driver_path = dir_path + '\\chromedriver.exe'
        driver = Chrome(options = options)
        google_reviews = extract_google_reviews(driver, 'MechaniGO.ph')
        fb_reviews = extract_fb_reviews("https://cache1.phantombooster.com/mltYLBVqs54/BrdC1EufIjjCsQ0dIhN4cQ/phantombuster_fbreviews.csv")
        df_reviews = pd.concat([google_reviews, fb_reviews]).dropna().sort_values('date', ascending=False).drop_duplicates().reset_index(drop=True)
        df_reviews = df_reviews[df_reviews['raw'] != ''].dropna().reset_index(drop=True)
        df_reviews['date'] = pd.to_datetime(df_reviews['date'])
        df_reviews.insert(1, 'month', df_reviews['date'].dt.month)
        df_reviews.insert(2, 'year', df_reviews['date'].dt.year)
        st.dataframe(df_reviews)
    finally:
        run_cond = True
        driver.quit()