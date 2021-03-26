# Data Science Portfolio

# Apply Machine Learning Algorithms in Predicting Stocks Return 
## Using Taiwanse stock market as an example



```python
import pandas as pd
import sqlalchemy 
import tejapi
import numpy as np
from datetime import date,timedelta
import matplotlib.pyplot as plt
import time
DATABASE_NAME='twse_stock_database'

stock_list=stock_list['coid'].to_list()
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 15)

```


```python
t1 = time.time()
close_price = pd.read_sql('台股收盤價',engine)
t2 = time.time()
print('Close price data loaded', round(t2 - t1,2), 'secs', '; Data shape:',close_price.shape)

financial_statement = pd.read_sql('台股財報',engine)
t3 = time.time()
print('Financial statement data loaded', round(t3 - t2,2), 'secs', '; Data shape:',financial_statement.shape)

stock_return = pd.read_sql('台股報酬率',engine)
t4 = time.time()
print('Stock return data loaded', round(t4 - t3,2), 'secs', '; Data shape:',stock_return.shape)

stock_vol = pd.read_sql('台股波動率',engine)
t5 = time.time()
print('Volatility data loaded', round(t5 - t4,2), 'secs', '; Data shape:',stock_vol.shape)

stock_beta = pd.read_sql('台股beta',engine)
t6 = time.time()
print('Beta data loaded', round(t6 - t5,2), 'secs', '; Data shape:',stock_beta.shape)

monthly_sales = pd.read_sql('台股月營收',engine)
t7 = time.time()
print('Monthly revenue loaded', round(t7 - t6,2), 'secs', '; Data shape:',monthly_sales.shape)

cash_flow = pd.read_sql('台股現金流',engine)
t8 = time.time()
print('Cash flow data loaded', round(t8 - t7,2), 'secs', '; Data shape:',cash_flow.shape)
print('All data loaded', round(t8 - t1,2), 'secs')
```
