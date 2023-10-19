import pandas as pd

from sqlalchemy import create_engine
from time import time 

engine = create_engine('postgresql://user:user@pgdatabase:5432/churn_data')

engine.connect()

df = pd.read_csv("dataset.csv")
df.head(0).to_sql(name= 'churn_data', con= engine, if_exists= 'fail')
df.to_sql(name = 'churn_data',con = engine,if_exists = 'replace')


    
    
