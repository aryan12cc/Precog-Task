'''
A script to scrape the dataset of the officially recognized 
countries and the top 500 most populated cities of the world.

Dataset 1 -- The officially recognized countries are taken from https://www.worldometers.info/geography/how-many-countries-are-there-in-the-world/
                State of Palestine and Holy See (Vatican City) is not included as they are not in the UN members list
Dataset 2 -- The top 500 most populated cities are taken from https://worldpopulationreview.com/cities
Dataset 3 -- Combined from dataset 1 and 2
'''

import pandas
import requests
from bs4 import BeautifulSoup

# function to scrape the data from https://www.worldometers.info/geography/how-many-countries-are-there-in-the-world/
# only UN member states are included as official countries in the dataset
def scrape_dataset1():
    # get the html content of the website
    url = 'https://www.worldometers.info/geography/how-many-countries-are-there-in-the-world/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # get the contents of the table containing all the countries
    country_table = soup.find('table', id = 'example2')
    # get the list of countries
    country_list = country_table.find_all('a')
    # make a csv file to store the data via pandas
    country_data = []
    for country in country_list:
        if country.text == 'Holy See' or country.text == 'State of Palestine':
            continue
        country_data.append(country.text)
    # save the data to a csv file
    df = pandas.DataFrame(country_data, columns = ['Name'])
    df.to_csv('datasets/countries_only.csv', index = False)

# function to scrape data from https://worldpopulationreview.com/cities
# only the top 500 cities are included
def scrape_dataset2():
    # get the html content of the website
    url = 'https://worldpopulationreview.com/cities'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # get the contents of the table containing the list of cities
    cities_table = soup.find('tbody', class_ = 'relative z-10 text-sm')
    # get the city names in all the rows of the table
    cities_list = cities_table.find_all('a', class_ = 'text-wpr-link')
    city_data = []
    for i, city in enumerate(cities_list):
        # we only need the top 500 most populous cities
        if i == 500:
            break
        city_data.append(city.text)
    # save the data to a csv file
    df = pandas.DataFrame(city_data, columns = ['Name'])
    df.to_csv('datasets/cities_only.csv', index = False)

def combine_datasets1_and_2():
    # read the csv files
    countries = pandas.read_csv('datasets/countries_only.csv')
    cities = pandas.read_csv('datasets/cities_only.csv')
    # combine the two datasets
    combined = pandas.concat([countries, cities])
    # save the data to a csv file
    combined.to_csv('datasets/countries_and_cities.csv', index = False)

def scrape_datasets():
    scrape_dataset1()
    scrape_dataset2()
    combine_datasets1_and_2()

if __name__ == '__main__':
    scrape_datasets()