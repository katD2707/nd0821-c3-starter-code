"""
Test the live prediction endpoint on Heroku
"""
import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")


features = {
   'age':50,
    'workclass':"Private", 
    'fnlgt':234721,
    'education':"Doctorate",
    'education_num':16,
    'marital_status':"Separated",
    'occupation':"Exec-managerial",
    'relationship':"Not-in-family",
    'race':"Black",
    'sex':"Female",
    'capital_gain':0,
    'capital_loss':0,
    'hours_per_week':50,
    'native_country':"United-States"
}


app_url = "https://blabla-d5t6.onrender.com/predict-income"

response = requests.post(app_url, json=features)

logging.info("Testing Heroku app")
logging.info(f"Status code: {response.status_code}")
logging.info(f"Response content: {response.json()}")