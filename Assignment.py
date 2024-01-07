#!/usr/bin/env python
# coding: utf-8

# #  Web scrapping of Properties data from real estate advertisement website using Python 

# In[1]:


import pandas as pd
import requests
from bs4 import BeautifulSoup as soup
import re
import json


# In[2]:


#header
header=({'user-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0','Accept-Language':'en-US,en;q=0.5'})


# In[3]:


url='https://www.magicbricks.com/property-for-sale/residential-real-estate?bedroom=&proptype=Multistorey-Apartment,Builder-Floor-Apartment,Penthouse,Studio-Apartment&cityName=Pune'


# In[4]:


req=requests.get(url=url,headers=header)


# In[5]:


req


# In[6]:


req.content


# In[7]:


B_soup=soup(req.text,'html.parser')
B_soup


# In[8]:


h2_elements = B_soup.find_all('h2')

property_list = [h2.get_text(strip=True) for h2 in h2_elements]
property_list


# In[9]:


price =B_soup.find_all("div", attrs={'class': 'mb-srp__card__price--amount'})
price


# In[10]:


num_properties=3000
property_titles = []
property_specs = []
property_floors = []
property_areas = []
property_values = []


# In[ ]:


page_num = 1
while len(property_titles) < num_properties:
        page_url = f'{url}&page={page_num}'
        req = requests.get(page_url, headers=header)
        B_soup = soup(req.content, 'html.parser')

        titles = [title.text.strip() for title in B_soup.find_all('h2', class_='pro-title')]
        specs = [spec.text.strip() for spec in B_soup.find_all('div', class_='pro-specifications')]
        floors = [floor.text.strip() for floor in B_soup.find_all('div', class_='pro-list-1')]
        areas = [area.text.strip() for area in B_soup.find_all('div', class_='pro-list-2')]
        values = [value.text.strip() for value in B_soup.find_all('span', class_='price')]

        property_titles.extend(titles)
        property_specs.extend(specs)
        property_floors.extend(floors)
        property_areas.extend(areas)
        property_values.extend(values)
        
        page_num += 1


# In[ ]:



    data = {'Title': property_titles[:num_properties],
            'Specification': property_specs[:num_properties],
            'Floor': property_floors[:num_properties],
            'Area': property_areas[:num_properties],
            'Value': property_values[:num_properties]}

    df = pd.DataFrame(data)
    df = df.drop_duplicates()

    return df

# URL for web
url = 'https://www.magicbricks.com/property-for-sale/residential-real-estate?bedroom=&proptype=Multistorey-Apartment,Builder-Floor-Apartment,Penthouse,Studio-Apartment&cityName=Pune'

# Scrape data from the website
property_data = scrape_properties(url, num_properties=3000)

# Save the data to a CSV file
property_data.to_csv('property_data.csv', index=False)


# above code is compiling whole day. so i just kept these above code to understand for you that what i did .

# # Developing AI based Property valuation model

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


# In[ ]:


# Load the scraped data
property_data = pd.read_csv('property_data.csv')


# In[ ]:


# Preprocess the data
le = LabelEncoder()
property_data['Specification'] = le.fit_transform(property_data['Specification'])
property_data['Floor'] = le.fit_transform(property_data['Floor'])


# In[ ]:


# Define input variables (X) and output variable (y)
X = property_data[['Specification', 'Floor', 'Area']]
y = property_data['Value']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


# Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)


# In[ ]:


# Random Forest model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)


# In[ ]:


# Decision Tree model
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)


# In[ ]:


# Evaluate the models
print(f'Linear Regression MSE: {mean_squared_error(y_test, linear_predictions)}')
print(f'Random Forest MSE: {mean_squared_error(y_test, rf_predictions)}')
print(f'Decision Tree MSE: {mean_squared_error(y_test, dt_predictions)}')


# same as above.These above code didn't run because web srapping is not completed. But I tried to show what code I wrote. 

# # Another way as below:

# In[13]:


import pandas as pd
df=pd.read_csv('Pune_House_Data.csv')
df


# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


# In[16]:


le = LabelEncoder()
df['location'] = le.fit_transform(df['location'])
df['size'] = le.fit_transform(df['size'])


# In[18]:


# Define input variables (X) and output variable (y)
X = df[['location', 'size', 'total_sqft']]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:




