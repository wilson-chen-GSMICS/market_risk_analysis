# Data Science Portfolio

# Data Science Project: Apply Machine Learning Algorithms in Predicting Stocks Return 
## Using Taiwanse stock market as an example



```python
import pandas as pd
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 15)
import sqlalchemy 
import tejapi
import numpy as np
from datetime import date,timedelta
import matplotlib.pyplot as plt
import time
DATABASE_NAME='tej_stock'
engine = sqlalchemy.create_engine('mysql+pymysql://root:Tej123456@localhost:3306/'+DATABASE_NAME)
tejapi.ApiConfig.api_key = "IbDhqxEj1XZy9J4W9uyVd3qYa945Ve"
stock_list =pd.read_excel("F://2019量化競賽/TEJ台股名單.xlsx")
stock_list=stock_list['coid'].to_list()
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

    C:\Users\Administrator\anaconda3\lib\site-packages\pymysql\cursors.py:170: Warning: (3719, "'utf8' is currently an alias for the character set UTF8MB3, but will be an alias for UTF8MB4 in a future release. Please consider using UTF8MB4 in order to be unambiguous.")
      result = self._query(query)
    

    Close price data loaded 155.08 secs ; Data shape: (2276389, 16)
    Financial statement data loaded 3.49 secs ; Data shape: (38695, 29)
    Stock return data loaded 78.08 secs ; Data shape: (1865397, 9)
    Volatility data loaded 84.84 secs ; Data shape: (1660669, 10)
    Beta data loaded 46.94 secs ; Data shape: (1819030, 4)
    Monthly revenue loaded 4.4 secs ; Data shape: (94489, 10)
    Cash flow data loaded 1.03 secs ; Data shape: (37379, 6)
    All data loaded 373.86 secs
    


```python
stock_return = stock_return.drop(columns=['index']).set_index('日期').sort_values(['證券碼','日期'],ascending=[True, True]).drop_duplicates()
close_price = close_price.drop(columns=['Date']).set_index('日期').sort_values(['證券碼','日期'],ascending=[True, True]).drop_duplicates()
stock_vol = stock_vol.drop(columns=['index']).set_index('日期').sort_values(['證券碼','日期'],ascending=[True, True]).drop_duplicates()
stock_beta = stock_beta.drop(columns=['index']).set_index('日期').sort_values(['證券碼','日期'],ascending=[True, True]).drop_duplicates()
monthly_sales = monthly_sales.sort_values(['證券碼','營收發布日','資料日期'],ascending=[True, True, True]).drop_duplicates(['證券碼', '營收發布日'],keep='last').set_index('營收發布日')

cash_flow = cash_flow.drop(columns=['index','公司','公司名'])
financial_statement['年月'] = pd.to_datetime(financial_statement['財務資料日']).dt.to_period('m')
cash_flow['年月'] = pd.to_datetime(cash_flow['年月']).dt.to_period('m')
financial_statement = financial_statement.merge(cash_flow, on= ['年月','證券碼'])
financial_statement = financial_statement.drop(columns=['Date'])
financial_statement = financial_statement.sort_values(['證券碼','財務資料日','財報發布日'],ascending=[True, True, True]).drop_duplicates()
financial_statement = financial_statement.drop_duplicates(['證券碼', '財報發布日'],keep='last')
```


```python
data_list = [('Stock price data',close_price), ('Financial statement data',financial_statement), ('Stock return data',stock_return), ('Stock volatility',stock_vol), ('Stock beta', stock_beta), 
             ('Monthly return',monthly_sales), ('Cash flow data',cash_flow)]
for data in data_list:
    data[1].head(1)
```


```python
data_list = [('Stock price',close_price), ('Stock return',stock_return), ('Stock volatility',stock_vol),('Stock beta', stock_beta),  ('Monthly return',monthly_sales)]

def data_check_plot(df):
    df
    df1=df.groupby(df.index)['證券碼'].count()
    return df1

fig, axes = plt.subplots(2,2,figsize=(16,10))
for data, ax in zip(data_list,axes.flat):
    ax.plot(data_check_plot(data[1]))
    ax.set_title(data[0])
```


![png](output_5_0.png)


1. 


```python
pd.DataFrame(close_price.groupby(close_price.index)['證券碼'].count().sort_values().head(10))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>證券碼</th>
    </tr>
    <tr>
      <th>日期</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-08-15</th>
      <td>1063</td>
    </tr>
    <tr>
      <th>2020-03-10</th>
      <td>1147</td>
    </tr>
    <tr>
      <th>2016-07-06</th>
      <td>1174</td>
    </tr>
    <tr>
      <th>2018-08-07</th>
      <td>1189</td>
    </tr>
    <tr>
      <th>2017-07-05</th>
      <td>1220</td>
    </tr>
    <tr>
      <th>2017-06-27</th>
      <td>1223</td>
    </tr>
    <tr>
      <th>2019-12-13</th>
      <td>1235</td>
    </tr>
    <tr>
      <th>2020-02-12</th>
      <td>1318</td>
    </tr>
    <tr>
      <th>2018-06-25</th>
      <td>1326</td>
    </tr>
    <tr>
      <th>2017-06-19</th>
      <td>1348</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame(stock_beta.groupby(stock_beta.index)['證券碼'].count().sort_values().head(10))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>證券碼</th>
    </tr>
    <tr>
      <th>日期</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-09-08</th>
      <td>31</td>
    </tr>
    <tr>
      <th>2020-09-07</th>
      <td>31</td>
    </tr>
    <tr>
      <th>2016-12-28</th>
      <td>65</td>
    </tr>
    <tr>
      <th>2018-08-15</th>
      <td>1016</td>
    </tr>
    <tr>
      <th>2020-03-10</th>
      <td>1125</td>
    </tr>
    <tr>
      <th>2016-07-06</th>
      <td>1145</td>
    </tr>
    <tr>
      <th>2018-08-07</th>
      <td>1163</td>
    </tr>
    <tr>
      <th>2017-07-05</th>
      <td>1175</td>
    </tr>
    <tr>
      <th>2017-06-27</th>
      <td>1189</td>
    </tr>
    <tr>
      <th>2019-12-13</th>
      <td>1201</td>
    </tr>
  </tbody>
</table>
</div>




```python
missed_beta = stock_beta[:'2020-09-08']
date_set = missed_beta.index.sort_values().unique()
select_date = date_set[-240:]

```


```python
def cal_beta(df):
    np_array = df[['index_ROI','stock_ROI']].values
    m = np_array[:,0] # market returns are column zero from numpy array
    s = np_array[:,1] # stock returns are column one from numpy array
    covariance = np.cov(s,m) # Calculate covariance between stock and market
    beta = covariance[0,1]/covariance[1,1]
    return beta

def beta_benchmark(df):
    t0 = time.time()
    lower_bound_date = (df[0] - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    upper_bound_date = (df[-1] + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    mdate={'gt':lower_bound_date,'lt':upper_bound_date}
    TW_Index = tejapi.get('TWN/EWIPRCD',idx_id='IX0001' ,mdate=mdate,chinese_column_name=True,paginate=True).sort_values('日期')
    TW_Index['日期'] = pd.to_datetime(TW_Index['日期'].dt.date)
    TW_Index['index_ROI'] = np.log(TW_Index['指數收盤價']) - np.log(TW_Index['指數收盤價'].shift(1))
    TW_Index = TW_Index.set_index('日期')
    close_price_selected = close_price[['證券碼','收盤價-除權息']]
    close_price_selected['stock_ROI'] = close_price_selected.groupby('證券碼')['收盤價-除權息'].apply(lambda x: np.log(x) - np.log(x.shift(1)))
    TW_Beta = close_price_selected.merge(TW_Index['index_ROI'], left_index=True, right_index=True)
    TW_Beta = TW_Beta.dropna()
    t1 = time.time()
    print("Preprocess finished",t1 - t0, 'secs')
    return TW_Beta
```


```python
bt = beta_benchmark(select_date)
bt
```

    C:\Users\Administrator\anaconda3\lib\site-packages\ipykernel_launcher.py:19: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    

    Preprocess finished 2.826096534729004 secs
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>證券碼</th>
      <th>收盤價-除權息</th>
      <th>stock_ROI</th>
      <th>index_ROI</th>
    </tr>
    <tr>
      <th>日期</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-09-16</th>
      <td>1101</td>
      <td>36.1331</td>
      <td>0.001246</td>
      <td>0.006497</td>
    </tr>
    <tr>
      <th>2019-09-16</th>
      <td>1102</td>
      <td>41.3828</td>
      <td>0.010221</td>
      <td>0.006497</td>
    </tr>
    <tr>
      <th>2019-09-16</th>
      <td>1103</td>
      <td>16.7148</td>
      <td>0.011360</td>
      <td>0.006497</td>
    </tr>
    <tr>
      <th>2019-09-16</th>
      <td>1104</td>
      <td>17.8728</td>
      <td>0.002650</td>
      <td>0.006497</td>
    </tr>
    <tr>
      <th>2019-09-16</th>
      <td>1108</td>
      <td>6.9192</td>
      <td>0.000000</td>
      <td>0.006497</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-09-08</th>
      <td>9951</td>
      <td>85.0000</td>
      <td>0.019003</td>
      <td>0.004921</td>
    </tr>
    <tr>
      <th>2020-09-08</th>
      <td>9955</td>
      <td>19.5000</td>
      <td>0.012903</td>
      <td>0.004921</td>
    </tr>
    <tr>
      <th>2020-09-08</th>
      <td>9958</td>
      <td>127.0000</td>
      <td>-0.042396</td>
      <td>0.004921</td>
    </tr>
    <tr>
      <th>2020-09-08</th>
      <td>9960</td>
      <td>28.5000</td>
      <td>0.019487</td>
      <td>0.004921</td>
    </tr>
    <tr>
      <th>2020-09-08</th>
      <td>9962</td>
      <td>9.3000</td>
      <td>-0.005362</td>
      <td>0.004921</td>
    </tr>
  </tbody>
</table>
<p>404422 rows × 4 columns</p>
</div>




```python
beta_append = bt.groupby('證券碼').apply(cal_beta())
beta_append
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-165-3783b9236d5a> in <module>
    ----> 1 beta_append = bt.groupby('證券碼').apply(cal_beta())
          2 beta_append
    

    TypeError: cal_beta() missing 1 required positional argument: 'df'



```python
stock_beta
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>證券碼</th>
      <th>Beta-240</th>
    </tr>
    <tr>
      <th>日期</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-12-28</th>
      <td>1101</td>
      <td>0.887393</td>
    </tr>
    <tr>
      <th>2015-12-29</th>
      <td>1101</td>
      <td>0.893812</td>
    </tr>
    <tr>
      <th>2015-12-30</th>
      <td>1101</td>
      <td>0.895472</td>
    </tr>
    <tr>
      <th>2015-12-31</th>
      <td>1101</td>
      <td>0.903102</td>
    </tr>
    <tr>
      <th>2016-01-04</th>
      <td>1101</td>
      <td>0.921675</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-09-18</th>
      <td>9962</td>
      <td>0.738622</td>
    </tr>
    <tr>
      <th>2020-09-21</th>
      <td>9962</td>
      <td>0.737519</td>
    </tr>
    <tr>
      <th>2020-09-22</th>
      <td>9962</td>
      <td>0.737799</td>
    </tr>
    <tr>
      <th>2020-09-23</th>
      <td>9962</td>
      <td>0.737128</td>
    </tr>
    <tr>
      <th>2020-09-24</th>
      <td>9962</td>
      <td>0.724843</td>
    </tr>
  </tbody>
</table>
<p>1811139 rows × 2 columns</p>
</div>




```python

```


```python
small = stock_return['2015-11-01':'2016-03-31']#.groupby(stock_return['2015-11-01':'2016-03-31'].index)['證券碼'].count()
```


```python
small['2016-01-04':'2016-01-04']
#.drop_duplicates().groupby(small.index)['證券碼'].count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>證券碼</th>
      <th>日報酬</th>
      <th>週報酬</th>
      <th>月報酬</th>
      <th>季報酬</th>
      <th>半年報酬</th>
      <th>年報酬</th>
    </tr>
    <tr>
      <th>日期</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-01-04</th>
      <td>1101</td>
      <td>-0.043040</td>
      <td>-0.057583</td>
      <td>-0.143991</td>
      <td>-0.326590</td>
      <td>-0.362971</td>
      <td>-0.403245</td>
    </tr>
    <tr>
      <th>2016-01-04</th>
      <td>1101</td>
      <td>0.000000</td>
      <td>-0.066568</td>
      <td>-0.148948</td>
      <td>-0.360491</td>
      <td>-0.366703</td>
      <td>-0.377807</td>
    </tr>
    <tr>
      <th>2016-01-04</th>
      <td>1102</td>
      <td>-0.042877</td>
      <td>-0.055569</td>
      <td>-0.083991</td>
      <td>-0.246860</td>
      <td>-0.260511</td>
      <td>-0.316675</td>
    </tr>
    <tr>
      <th>2016-01-04</th>
      <td>1102</td>
      <td>0.000000</td>
      <td>-0.057370</td>
      <td>-0.085741</td>
      <td>-0.277631</td>
      <td>-0.250837</td>
      <td>-0.304906</td>
    </tr>
    <tr>
      <th>2016-01-04</th>
      <td>1103</td>
      <td>-0.012688</td>
      <td>-0.021058</td>
      <td>-0.044730</td>
      <td>-0.091436</td>
      <td>-0.210730</td>
      <td>-0.393749</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2016-01-04</th>
      <td>9958</td>
      <td>0.000000</td>
      <td>-0.014970</td>
      <td>-0.002010</td>
      <td>0.082720</td>
      <td>0.094048</td>
      <td>-0.007563</td>
    </tr>
    <tr>
      <th>2016-01-04</th>
      <td>9960</td>
      <td>-0.005246</td>
      <td>0.010582</td>
      <td>-0.005246</td>
      <td>-0.048786</td>
      <td>-0.076882</td>
      <td>-0.131690</td>
    </tr>
    <tr>
      <th>2016-01-04</th>
      <td>9960</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.002636</td>
      <td>-0.031089</td>
      <td>-0.076882</td>
      <td>-0.131690</td>
    </tr>
    <tr>
      <th>2016-01-04</th>
      <td>9962</td>
      <td>0.023719</td>
      <td>0.016689</td>
      <td>-0.044514</td>
      <td>-0.228369</td>
      <td>-0.148419</td>
      <td>-0.447536</td>
    </tr>
    <tr>
      <th>2016-01-04</th>
      <td>9962</td>
      <td>0.000000</td>
      <td>0.015289</td>
      <td>-0.045831</td>
      <td>-0.226174</td>
      <td>-0.162586</td>
      <td>-0.451832</td>
    </tr>
  </tbody>
</table>
<p>2916 rows × 7 columns</p>
</div>




```python
###############################Drop#####################################################

cum_vol = close_price.groupby('證券碼')['成交量(千股)'].rolling(5).sum().reset_index(level=0)
cum_vol.rename(columns={'成交量(千股)':'5日累積成交量'}, inplace=True)

cum_vol['20日累積成交量'] = close_price.groupby('證券碼')['成交量(千股)'].rolling(20).sum().reset_index(level=0).drop(columns='證券碼')
cum_vol['前20日累積成交量'] = cum_vol['20日累積成交量'].shift(20)
cum_vol['前5日累積成交量'] = cum_vol['5日累積成交量'].shift(5)
cum_vol['20日成交量增長'] = (cum_vol['20日累積成交量']/cum_vol['前20日累積成交量'])-1
cum_vol['5日成交量增長'] = (cum_vol['5日累積成交量']/cum_vol['前5日累積成交量'] )-1

close_price = close_price.reset_index().merge(cum_vol[['證券碼', '20日成交量增長','5日成交量增長']].reset_index(),on=['日期','證券碼']).set_index('日期')
close_price['vol_growth'] = close_price[['證券碼','成交量(千股)']].groupby('證券碼').pct_change()
```


```python
###############################Pre-Processing#####################################################

list1=[ 'ROA(C) 稅前息前折舊前', 'ROE(A)-稅後', 'CFO/負債','CFO/合併總損益', '現金流量比率', 
       '營業毛利率', '營業利益率', '稅後淨利率', '營收成長率', '營業毛利成長率', '營業利益成長率']

test=financial_statement

for acc in list1:
    test[acc+'-四季平均']=test.groupby('證券碼')[acc].rolling(4).mean().reset_index(level=0, drop=True)
    

test['每股盈餘-四季累計']=test.groupby('證券碼')['每股盈餘'].rolling(4).sum().reset_index(level=0, drop=True)/4
       

financial_statement = test.set_index('財報發布日')


close_price['總市值'] = close_price['收盤價-除權息'] * close_price['流通股數(千股)']
close_price = close_price.dropna(subset=['20日成交量增長', '5日成交量增長', 'vol_growth'])
```


```python
def data_combination_and_preprocess():
    data_summary = close_price[['證券碼', '成交量(千股)', '總市值', '現金股利率', '收盤價-除權息','20日成交量增長', '5日成交量增長', 'vol_growth']]
    data_summary=pd.merge_asof(data_summary.sort_index()['2016-06-02':], \
                               financial_statement.sort_index()['2015-01-01':],left_index=True,right_index=True,by='證券碼')
    data_summary=pd.merge_asof(data_summary.sort_index()['2016-06-02':], \
                               stock_return[['證券碼', '日報酬', '週報酬', '月報酬', '季報酬', '半年報酬', '年報酬']].sort_index()\
                               ['2016-06-02':],left_index=True,right_index=True,by='證券碼')
    data_summary=pd.merge_asof(data_summary.sort_index()['2016-06-02':], \
                               stock_vol[['證券碼','rolling_std_20', 'rolling_std_60', 'rolling_std_10', 'rolling_std_120', 'Trend_20/120', 'Trend_10/20']]\
                               .sort_index()['2016-06-02':],left_index=True,right_index=True,by='證券碼')
    data_summary=pd.merge_asof(data_summary.sort_index()['2016-06-02':], \
                               stock_beta.sort_index()['2016-06-02':],left_index=True,right_index=True,by='證券碼')
    data_summary=pd.merge_asof(data_summary.sort_index()['2016-06-02':], \
                               monthly_sales[['證券碼', '單月營收成長率%', '移動12月累計營收年增率','移動12月累計營收']].sort_index()['2016-01-01':],\
                               left_index=True,right_index=True,by='證券碼')
    data_summary = data_summary.sort_values(['日期','證券碼'])
    data_summary.isna().sum()
    data_summary = data_summary.dropna(subset=['ROE(A)-稅後-四季平均', 'ROA(C) 稅前息前折舊前-四季平均', 'CFO/負債-四季平均',
           'CFO/合併總損益-四季平均', '現金流量比率-四季平均', '每股盈餘-四季累計', '營業毛利率-四季平均','營業利益率-四季平均', 
           '稅後淨利率-四季平均', '營收成長率-四季平均', '營業毛利成長率-四季平均',
           '營業利益成長率-四季平均'])
    data_summary = data_summary.dropna(subset=['日報酬', '週報酬', '月報酬', '季報酬', '半年報酬', '年報酬','rolling_std_20',
                                               'rolling_std_60', 'rolling_std_10','rolling_std_120', 'Trend_20/120', 'Trend_10/20',
                                               'Beta-240'])
    data_summary['Earning_to_price'] = data_summary['每股盈餘-四季累計'] / data_summary['收盤價-除權息']
    data_summary['Book_to_price'] = data_summary['每股淨值(F)-TSE公告數'] / data_summary['收盤價-除權息']
    data_summary['Sales_to_price'] = data_summary['移動12月累計營收'] / data_summary['總市值']
    data_summary['CF_to_price'] = data_summary['來自營運之現金流量'] / data_summary['總市值']

    data_summary['成交值'] = data_summary['成交量(千股)'] * data_summary['收盤價-除權息'] 
    data_summary['Sharpe-20days'] = data_summary['月報酬'] / data_summary['rolling_std_20']
    data_summary['Sharpe-60days'] = data_summary['季報酬'] / data_summary['rolling_std_60']
    data_summary['Sharpe-120days'] = data_summary['半年報酬'] / data_summary['rolling_std_120']
    data_summary=data_summary[['證券碼','收盤價-除權息', '成交量(千股)', '總市值', '現金股利率', 'ROA(C) 稅前息前折舊前', 'ROE(A)-稅後', '營業毛利率',
       '營業利益率', '稅後淨利率', 'CFO/負債', 'CFO/合併總損益', '現金流量比率', 
       '營收成長率', '營業毛利成長率', '營業利益成長率', '流動比率', '速動比率', '負債比率', '長期資金適合率(A)',
       '應收帳款週轉次數', '總資產週轉次數', '存貨週轉率(次)',  'ROA(C) 稅前息前折舊前-四季平均',
       'ROE(A)-稅後-四季平均', 'CFO/負債-四季平均', 'CFO/合併總損益-四季平均', '現金流量比率-四季平均',
       '營業毛利率-四季平均', '營業利益率-四季平均', '稅後淨利率-四季平均', '營收成長率-四季平均', '營業毛利成長率-四季平均',
       '營業利益成長率-四季平均', '日報酬', '週報酬', '月報酬', '季報酬', '半年報酬', '年報酬',
       'rolling_std_20', 'rolling_std_60', 'rolling_std_10', 'rolling_std_120', 'Trend_20/120', 'Trend_10/20','Beta-240','Sharpe-20days', 'Sharpe-60days',
       'Sharpe-120days','Earning_to_price','Book_to_price','Sales_to_price','CF_to_price', '單月營收成長率%',
       '移動12月累計營收年增率','20日成交量增長', '5日成交量增長', 'vol_growth']]
    data_summary['持有一日收益']=data_summary.groupby(['證券碼'])['日報酬'].shift(-2)
    data_summary['持有五日收益']=data_summary.groupby(['證券碼'])['週報酬'].shift(-6)
    data_summary['持有二十日收益']=data_summary.groupby(['證券碼'])['月報酬'].shift(-21)
    data_summary['持有六十日收益']=data_summary.groupby(['證券碼'])['季報酬'].shift(-61)
    data_summary['持有二十日夏普']=data_summary.groupby(['證券碼'])['Sharpe-20days'].shift(-21)
    data_summary['持有六十日夏普']=data_summary.groupby(['證券碼'])['Sharpe-60days'].shift(-61)
    return data_summary
data_summary = data_combination_and_preprocess()
```


```python
no_na_data = data_summary.groupby(data_summary.index).apply(lambda x: x.fillna(x.median()))
no_na_data=no_na_data.droplevel(level=0)
no_na_data = no_na_data.fillna(0)
```

    C:\Users\Administrator\anaconda3\lib\site-packages\numpy\lib\nanfunctions.py:995: RuntimeWarning: All-NaN slice encountered
      result = np.apply_along_axis(_nanmedian1d, axis, a, overwrite_input)
    


```python
def data_standard(data): 
    columns_to_exclude = ['證券碼','持有一日收益', '持有五日收益','持有二十日收益','持有六十日收益','持有二十日夏普','持有六十日夏普','總市值',
                          '收盤價-除權息', '日報酬', '週報酬', '月報酬', '季報酬','半年報酬', '年報酬',
                          'rolling_std_20', 'rolling_std_60', 'rolling_std_10', 'rolling_std_120',
                          'Beta-240']
    new_df = pd.DataFrame()
    new_df[columns_to_exclude] = data[columns_to_exclude]
    columns_to_exclude = ['證券碼','持有一日收益', '持有五日收益','持有二十日收益','持有六十日收益','持有二十日夏普','持有六十日夏普',
                          '收盤價-除權息',
                          '日報酬', '週報酬', '月報酬', '季報酬','半年報酬', '年報酬','rolling_std_20', 'rolling_std_60',
                          'rolling_std_10', 'rolling_std_120',
                          'Beta-240']
    X_col = data.columns.drop(columns_to_exclude)
    for label in X_col:
        new_df[label+' z_score'] = data.groupby(data.index)[label].apply(lambda x: (x - x.mean())/x.std())
    print('Normalization completed.')
    return new_df
```


```python
NFG=data_standard(no_na_data)
NFG = NFG.drop(columns='vol_growth z_score')
eng_columns = ['id', 'target_1day', 'target_5day','target_20day', 'target_60day' ,'target_20sharpe','target_60sharpe', 'Market Cap','close',
               'mom_d', 'mom_w', 'mom_m', 'mom_q', 'mom_h', 'mom_y', 'rolling_std_20', 'rolling_std_60', 'rolling_std_10',
               'rolling_std_120','Beta-240', 'Sharpe-20days', 'Sharpe-60days',
               'Sharpe-120days', 'Trading Volume z_score', 'Market Cap z_score', 'Div Yield z_score',
               'ROA(C)  z_score', 'ROE(A) z_score', 'GP% z_score',
               'OP% z_score', 'NP% z_score', 'CFO/Debt z_score', 'CFO/Total_NI z_score',
               'CF_Ratio z_score', 'Rev_Growth z_score', 'GP_Growth z_score', 'NP_Growth z_score',
               'Liquidity z_score', 'Acid_test z_score', 'Debt ratop z_score', 'LT Capital Rate(A) z_score',
               'AR_Turn z_score', 'Assets_Turn z_score', 'Inv_Turn z_score',
                'ROA(C)-4 z_score', 'ROE(A)-4 z_score',
               'CFO/Debt-4 z_score', 'CFO/Total_NI-4 z_score', 'CF_Ratio-4 z_score',
               'GP-4 z_score', 'OP-4 z_score', 'NP-4 z_score',
               'Rev_Growth -4 z_score', 'GP Growth-4 z_score', 'NP Growth-4 z_score',
               'Trend_20/120 z_score', 'Trend_10/20 z_score','Earning_to_price z_score', 'Book_to_price z_score',
               'Sales_to_price z_score', 'CF_to_price z_score', '1 Month Sales Growth% z_score', 
               '12 Month Sales Growth z_score','20day_trade_vol z_score', '5day_trade_vol z_score'
               ]
NFG=NFG.dropna()
NFG.index.names = ['Date']
NFG.columns = eng_columns
```

    Normalization completed.
    


```python
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb

models = [('CatBoost',CatBoostRegressor(n_estimators = 300, loss_function = 'MAE',eval_metric = 'RMSE',silent =True)),
          ('LightGBM',lgb.LGBMRegressor(objective='regression',num_leaves=120,learning_rate=0.05, n_estimators=200)),
          ('XGB',xgb.XGBRegressor(learning_rate=0.05, max_depth=8, tree_method='gpu_hist', gpu_id=0,n_estimators=200,gamma=0.25,reg_alpha=0.25, reg_lambda=0.25))]

targets = ['target_20day', 'target_20sharpe']

end_date ='2019-03-31'
start_date = '2019-04-01'
Train=NFG[:end_date]
print ('Training set shape:',Train.shape)
Test=NFG[start_date:]
print ('Testing set shape:',Test.shape)
Test=Test.fillna(0)
Prediction_List=Test[['id', 'target_1day', 'target_5day','target_20day','target_60day', 'Market Cap','close','rolling_std_20']]
Prediction_List.columns =['證券碼', '持有一日收益', '持有五日收益', '持有二十日收益','持有六十日收益','總市值' ,'收盤價-除權息','二十日波動率']

import itertools
import time
for model, target in itertools.product(models,targets):
    X_Train = Train.drop(columns=['id', 'target_1day', 'target_5day','target_20day','target_20sharpe', 'target_60day','target_60sharpe', 'close','Market Cap'])
    Y_Train = Train[target]
    X_Test = Test.drop(columns=['id', 'target_1day', 'target_5day','target_20day','target_20sharpe', 'target_60day', 'target_60sharpe','close','Market Cap'])
    Y_Test = Test[target]
    reg = model[1]
    start = time.time()
    reg.fit(X_Train, Y_Train)
    end = time.time()
    Prediction_List[model[0]+'('+target+')'] = model[1].predict(X_Test)
    print(model[0],target," Training time: ", round(end - start,2), 'secs')
```

    Training set shape: (1011193, 65)
    Testing set shape: (574026, 65)
    

    C:\Users\Administrator\anaconda3\lib\site-packages\ipykernel_launcher.py:31: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    

    CatBoost target_20day 訓練時間:  88.29 secs
    CatBoost target_20sharpe 訓練時間:  83.58 secs
    LightGBM target_20day 訓練時間:  40.72 secs
    LightGBM target_20sharpe 訓練時間:  62.82 secs
    XGB target_20day 訓練時間:  28.43 secs
    XGB target_20sharpe 訓練時間:  30.76 secs
    


```python
Prediction_List.index.names = ['日期']
name_list = tejapi.get('TWN/EWNPRCSTD',coid=stock_list, chinese_column_name=True,paginate=True)
Big_Stock = Prediction_List.reset_index().merge(close_price[['證券碼','成交量(千股)']].reset_index(),on=['日期','證券碼'])
Big_Stock['總市值排名'] = Big_Stock['總市值'].groupby(Big_Stock['日期']).rank(ascending = False)
Big_Stock['成交量排名'] = Big_Stock['成交量(千股)'].groupby(Big_Stock['日期']).rank(ascending = False)
Big_Stock = Big_Stock.loc[(Big_Stock['成交量排名']<=1000)]
Big_Stock = Big_Stock.merge(name_list[['證券碼','上市別','證券名稱','TSE產業名' ]], right_on='證券碼', left_on='證券碼').set_index('日期')
```


```python
strategy_list = [str(model[0])+'('+target+')' for model, target in itertools.product(models,targets)]
groups = Big_Stock.reset_index()
for strategy in strategy_list:
    GP=Big_Stock.groupby(Big_Stock.index).apply(lambda x: x.sort_values(strategy, ascending=False))
    GP=GP.droplevel(level=0)
    GP[strategy+'_Group'] = GP.groupby(GP.index)[strategy].transform(lambda x: pd.qcut(x.rank(method='first'),50,labels=False))+1
    GP[strategy+'_Group'] = GP[strategy+'_Group'].transform(lambda x: int(x))
    GP = GP.reset_index()
    groups = groups.merge(GP[['日期','證券碼',strategy+'_Group']], on= ['日期','證券碼'])

groups = groups.set_index('日期').sort_index()
```


```python
def month_end_index(df):
    month_end_index = df.loc[df.groupby(df.index.to_period('M')).apply(lambda x: x.index.max())].index.unique()
    return month_end_index

def week_end_index(df):
    week_end_index = df.loc[df.groupby(df.index.to_period('W')).apply(lambda x: x.index.max())].index.unique()
    return week_end_index

def get_weight(df):
    print('calculating weights')
    df = df.reset_index()
    df['price_weighted'] = df.groupby(['日期','Group']).apply(lambda x: x['收盤價-除權息'] / x['收盤價-除權息'].sum()).droplevel(level=0).droplevel(level=0)
    df['cap_weighted'] = df.groupby(['日期','Group']).apply(lambda x: x['總市值'] / x['總市值'].sum()).droplevel(level=0).droplevel(level=0)
    df['simple_average'] = 1/20
    df = df.set_index('日期')
    return df
```


```python
def portfolio_resample(df):
    freq_list = ['D','W','M']
    daily_return = df[['證券碼','證券名稱','持有一日收益','總市值', '收盤價-除權息', '成交量(千股)', '上市別']]
    w_index = df.loc[df.groupby(df.index.to_period(freq_list[1])).apply(lambda x: x.index.min())].index.unique()  
    m_index = df.loc[df.groupby(df.index.to_period(freq_list[2])).apply(lambda x: x.index.min())].index.unique()  
    week_port = df[df.index.isin(w_index)]
    month_port = df[df.index.isin(m_index)]   
    cols = ['證券碼',
            'CatBoost(target_20day)', 'CatBoost(target_20sharpe)','LightGBM(target_20day)','LightGBM(target_20sharpe)','XGB(target_20day)', 'XGB(target_20sharpe)',
            'CatBoost(target_20day)_Group','CatBoost(target_20sharpe)_Group', 'LightGBM(target_20day)_Group','LightGBM(target_20sharpe)_Group', 
            'XGB(target_20day)_Group','XGB(target_20sharpe)_Group'
           ]
    return_w = pd.merge_asof(daily_return,week_port[cols],left_index=True,right_index=True, by=['證券碼'])
    return_m = pd.merge_asof(daily_return,month_port[cols],left_index=True,right_index=True, by=['證券碼']) 
    return return_w, return_m
```


```python
week_holding, month_holding = portfolio_resample(groups)
```


```python
def rank_by_strategy2(df,strategy):
    columns = ['證券碼','證券名稱',  '持有一日收益', '總市值', '收盤價-除權息', '成交量(千股)', '上市別',  strategy+'_Group']
    back_test_result = df.sort_values(['日期',strategy],ascending=[True,False])[columns]
    back_test_result = back_test_result.rename(columns ={strategy+'_Group':'Group'})
    return back_test_result

def weighted_return(df,freq,transaction_cost,strategy):
    freq_dict = {'D':', daily PnL (D)','W':', daily PnL(W)',"M":', daily PnL(M)'}
    return_table = pd.DataFrame(index=df.index.unique())
    weight_basis = ['simple_average','cap_weighted','price_weighted']
    df = get_weight(df) 
    print(strategy)
    for wb in weight_basis:
        df1 = df
        df1 = df1.reset_index().groupby(['日期','Group']).apply(lambda x:(x[wb] * x['持有一日收益']).sum()).unstack()
        df1 = df1[[1,50]]
        cols = [strategy+' Short position'+freq_dict[freq]+'-'+wb , strategy+' Long position'+freq_dict[freq]+'-'+wb]
        df1.columns = cols
        df1[strategy+', Long position-'+wb] = (((df1[strategy+' Long position'+freq_dict[freq]+'-'+wb]+1)*(1-transaction_cost)).cumprod()-1)
        df1[strategy+', Short position-'+wb] = (((df1[strategy+' Short position'+freq_dict[freq]+'-'+wb]+1)*(1-transaction_cost)).cumprod()-1)
        df1.loc[df1.index.min()] = 0  
        return_table = return_table.merge(df1,right_index=True, left_index=True)
    return return_table
```


```python
#New Test
t0 = time.time()
group_return_summary = pd.DataFrame(index =week_holding.index.unique())
for strategy in strategy_list:
    back_test_result = rank_by_strategy2(week_holding,strategy)
    group_return = weighted_return(back_test_result,'W',0.00,strategy)
    group_return_summary = group_return_summary.merge(group_return, right_index=True, left_index=True)
    
def cols_filter(df, keyword1):
    filtered_col = [col for col in df.columns if keyword1 in col]
    return filtered_col

long_table = group_return_summary[cols_filter(group_return_summary,', Long position-')]
long_table
t1 = time.time()
print ('Time consumed:',round(t1-t0,2),'secs')
```

    calculating weights
    CatBoost(target_20day)
    calculating weights
    CatBoost(target_20sharpe)
    calculating weights
    LightGBM(target_20day)
    calculating weights
    LightGBM(target_20sharpe)
    calculating weights
    XGB(target_20day)
    calculating weights
    XGB(target_20sharpe)
    Time consumed: 273.15 secs
    


```python
short_table = group_return_summary[cols_filter(group_return_summary,', Short position-')]
short_table.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CatBoost(target_20day), Short position-simple_average</th>
      <th>CatBoost(target_20day), Short position-cap_weighted</th>
      <th>CatBoost(target_20day), Short position-price_weighted</th>
      <th>CatBoost(target_20sharpe), Short position-simple_average</th>
      <th>CatBoost(target_20sharpe), Short position-cap_weighted</th>
      <th>CatBoost(target_20sharpe), Short position-price_weighted</th>
      <th>LightGBM(target_20day), Short position-simple_average</th>
      <th>...</th>
      <th>LightGBM(target_20sharpe), Short position-price_weighted</th>
      <th>XGB(target_20day), Short position-simple_average</th>
      <th>XGB(target_20day), Short position-cap_weighted</th>
      <th>XGB(target_20day), Short position-price_weighted</th>
      <th>XGB(target_20sharpe), Short position-simple_average</th>
      <th>XGB(target_20sharpe), Short position-cap_weighted</th>
      <th>XGB(target_20sharpe), Short position-price_weighted</th>
    </tr>
    <tr>
      <th>日期</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-04-01</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2019-04-02</th>
      <td>0.019732</td>
      <td>0.021797</td>
      <td>0.019557</td>
      <td>0.022420</td>
      <td>0.020790</td>
      <td>0.021567</td>
      <td>-0.000166</td>
      <td>...</td>
      <td>0.00523</td>
      <td>0.004237</td>
      <td>0.018931</td>
      <td>0.009267</td>
      <td>0.001182</td>
      <td>-0.001742</td>
      <td>0.001943</td>
    </tr>
    <tr>
      <th>2019-04-03</th>
      <td>0.006698</td>
      <td>0.002953</td>
      <td>0.004892</td>
      <td>0.017003</td>
      <td>0.007739</td>
      <td>0.009694</td>
      <td>-0.020902</td>
      <td>...</td>
      <td>0.00298</td>
      <td>-0.005786</td>
      <td>0.012649</td>
      <td>0.009232</td>
      <td>-0.006279</td>
      <td>-0.001426</td>
      <td>-0.001749</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 18 columns</p>
</div>




```python
daily_return_summary = group_return_summary[cols_filter(group_return_summary,'PnL')]
```


```python
import os
file_path = os. getcwd()
plt.style.use('seaborn-white')
def plot_return(df,name):
    weight_basis = ['simple_average','cap_weighted','price_weighted']
    for weight in weight_basis:
        cols = [col for col in df.columns if weight in col]
        fig,ax = plt.subplots(1,1,figsize=(16,10))
        for label in cols:
            ax.plot(df[label])
            x = df[label].index
            y = df[label]
            stg_return =', ' +str(round((y[-1]*100),2))+'%'
            ax.annotate(label+stg_return,
              xy     = (x[-1]+ timedelta(days=0), y[-1]),
              xytext = (x[-1], y[-1]))
            ax.spines['bottom'].set_color('#b0abab')
            ax.spines['top'].set_color('#b0abab')
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.grid(linestyle='--', alpha=0.5)
            ax.set_title('Backtest return-'+weight, fontsize=16, fontweight=1)
        plt.savefig(file_path+'//'+name+'-'+weight+'.png', dpi=400,bbox_inches='tight')
```


```python
plot_return(long_table,'strategy comparison-long position')
```


![png](output_34_0.png)



![png](output_34_1.png)



![png](output_34_2.png)



```python
plot_return(short_table,'strategy comparison-short position')
```


![png](output_35_0.png)



![png](output_35_1.png)



![png](output_35_2.png)



```python
def Simple_performanceEvaluation(ret,tsc,model):
    riskfreerate=0.03
    if not isinstance(ret, pd.Series):
        raise TypeError('You should input a pandas Series')
    days=240
    length = len(ret)
    rbar   = ret.mean()
    s      = ret.std()
    tstat  = np.sqrt(length)*rbar/s
    semivar = np.mean((np.where(ret-rbar<=0, ret-rbar, 0)) ** 2)
    downrisk = np.sqrt(semivar)
    sharperatio   = np.sqrt(days)*(rbar - riskfreerate/days)/s
    sortinoratio  = np.sqrt(days)*(rbar - riskfreerate/days)/downrisk
    transaction_cost=tsc    
    cumuret = ((ret + 1)*(1-transaction_cost)).cumprod()-1
    #cumuret = ((ret + 1)).cumprod()-1
    cumuret_per=cumuret*100
    cumuret_per=cumuret*100
    nav=cumuret+1
    waterlevel = nav.cummax()
    drawdown = (waterlevel-nav)/waterlevel
    geomean = (1 + cumuret[-1:]) ** (1 / length)
    cumuretann = (geomean ** days - 1)*100
    duration = np.zeros_like(drawdown)

    for k in range(length - 1):
        if drawdown[k + 1] > 0:
            duration[k + 1] = duration[k] + 1
        else:
            duration[k + 1] = 0

    mdd = drawdown.max()*100
    mddd = duration.max()
    perf=pd.DataFrame()
    feature_idx=['Number of observations:','Risk free rate:','Transaction Cost','Mean return (daily):',
                 '   t-statistics:','Standard deviation(daily):','Semi-variance (daily):','Down-side risk (daily):',
                 'Sharpe ratio (annualized):','Sortino ratio (annualized):','Cumulative return:','Geometric mean of 1+r:',
                 'Annualized return(Cummulative):','Maximum drawdown rate:','Maximum drawdown duration:']
    pef_summary=[length,riskfreerate,'%0.4f'%tsc,'%0.4f'%rbar,
                 '%0.4f'%tstat,'%0.4f'%s,'%0.4f'%semivar,'%0.4f'%downrisk,
                 '%0.4f'%sharperatio,'%0.4f'%sortinoratio,'%0.2f'%cumuret_per[-1:]+'%','%0.4f'%geomean,'%0.2f'%cumuretann+'%',
                 '%0.2f'%mdd+'%', '%d'%mddd+' days']
    perf['Performance/Algorithm']=feature_idx
    perf[model]=pef_summary

    return perf
```


```python
cols = cols_filter(group_return_summary,'PnL')
feature_idx=['Number of observations:','Risk free rate:','Transaction Cost','Mean return (daily):',
             '   t-statistics:','Standard deviation(daily):','Semi-variance (daily):','Down-side risk (daily):',
             'Sharpe ratio (annualized):','Sortino ratio (annualized):','Cumulative return:','Geometric mean of 1+r:',
             'Annualized return(Cummulative):','Maximum drawdown rate:','Maximum drawdown duration:']
perf_table=pd.DataFrame()
perf_table['Performance/Algorithm']=feature_idx

for label in cols:
    perf = Simple_performanceEvaluation(group_return_summary[label],0.00,label)
    perf_table = perf_table.merge(perf,on='Performance/Algorithm')
perf_table = perf_table.set_index('Performance/Algorithm')
```


```python
pd.options.display.max_rows = 999
Trans = perf_table.T
Trans.sort_values('Sharpe ratio (annualized):',ascending=False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Performance/Algorithm</th>
      <th>Number of observations:</th>
      <th>Risk free rate:</th>
      <th>Transaction Cost</th>
      <th>Mean return (daily):</th>
      <th>t-statistics:</th>
      <th>Standard deviation(daily):</th>
      <th>Semi-variance (daily):</th>
      <th>Down-side risk (daily):</th>
      <th>Sharpe ratio (annualized):</th>
      <th>Sortino ratio (annualized):</th>
      <th>Cumulative return:</th>
      <th>Geometric mean of 1+r:</th>
      <th>Annualized return(Cummulative):</th>
      <th>Maximum drawdown rate:</th>
      <th>Maximum drawdown duration:</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LightGBM(target_20day) Long positionDaily PnL(W)-simple_average</th>
      <td>366</td>
      <td>0.03</td>
      <td>0.0000</td>
      <td>0.0016</td>
      <td>1.8273</td>
      <td>0.0163</td>
      <td>0.0002</td>
      <td>0.0130</td>
      <td>1.3606</td>
      <td>1.7055</td>
      <td>68.07%</td>
      <td>1.0014</td>
      <td>40.56%</td>
      <td>39.49%</td>
      <td>82 days</td>
    </tr>
    <tr>
      <th>LightGBM(target_20day) Long positionDaily PnL(W)-cap_weighted</th>
      <td>366</td>
      <td>0.03</td>
      <td>0.0000</td>
      <td>0.0016</td>
      <td>1.6387</td>
      <td>0.0187</td>
      <td>0.0002</td>
      <td>0.0141</td>
      <td>1.2234</td>
      <td>1.6197</td>
      <td>68.45%</td>
      <td>1.0014</td>
      <td>40.77%</td>
      <td>40.54%</td>
      <td>87 days</td>
    </tr>
    <tr>
      <th>XGB(target_20sharpe) Long positionDaily PnL(W)-simple_average</th>
      <td>366</td>
      <td>0.03</td>
      <td>0.0000</td>
      <td>0.0010</td>
      <td>1.5804</td>
      <td>0.0118</td>
      <td>0.0001</td>
      <td>0.0089</td>
      <td>1.1151</td>
      <td>1.4663</td>
      <td>39.12%</td>
      <td>1.0009</td>
      <td>24.17%</td>
      <td>29.66%</td>
      <td>81 days</td>
    </tr>
    <tr>
      <th>LightGBM(target_20sharpe) Long positionDaily PnL(W)-simple_average</th>
      <td>366</td>
      <td>0.03</td>
      <td>0.0000</td>
      <td>0.0008</td>
      <td>1.2379</td>
      <td>0.0125</td>
      <td>0.0001</td>
      <td>0.0098</td>
      <td>0.8478</td>
      <td>1.0841</td>
      <td>30.68%</td>
      <td>1.0007</td>
      <td>19.18%</td>
      <td>34.94%</td>
      <td>84 days</td>
    </tr>
    <tr>
      <th>LightGBM(target_20day) Long positionDaily PnL(W)-price_weighted</th>
      <td>366</td>
      <td>0.03</td>
      <td>0.0000</td>
      <td>0.0009</td>
      <td>0.9181</td>
      <td>0.0191</td>
      <td>0.0002</td>
      <td>0.0147</td>
      <td>0.6418</td>
      <td>0.8338</td>
      <td>30.69%</td>
      <td>1.0007</td>
      <td>19.19%</td>
      <td>40.38%</td>
      <td>89 days</td>
    </tr>
    <tr>
      <th>XGB(target_20day) Long positionDaily PnL(W)-simple_average</th>
      <td>366</td>
      <td>0.03</td>
      <td>0.0000</td>
      <td>0.0007</td>
      <td>0.8241</td>
      <td>0.0163</td>
      <td>0.0002</td>
      <td>0.0129</td>
      <td>0.5486</td>
      <td>0.6915</td>
      <td>23.11%</td>
      <td>1.0006</td>
      <td>14.61%</td>
      <td>39.68%</td>
      <td>89 days</td>
    </tr>
    <tr>
      <th>CatBoost(target_20day) Short positionDaily PnL(W)-simple_average</th>
      <td>366</td>
      <td>0.03</td>
      <td>0.0000</td>
      <td>0.0007</td>
      <td>0.7718</td>
      <td>0.0173</td>
      <td>0.0002</td>
      <td>0.0137</td>
      <td>0.5133</td>
      <td>0.6496</td>
      <td>22.18%</td>
      <td>1.0005</td>
      <td>14.04%</td>
      <td>39.37%</td>
      <td>291 days</td>
    </tr>
    <tr>
      <th>XGB(target_20sharpe) Long positionDaily PnL(W)-price_weighted</th>
      <td>366</td>
      <td>0.03</td>
      <td>0.0000</td>
      <td>0.0005</td>
      <td>0.8269</td>
      <td>0.0122</td>
      <td>0.0001</td>
      <td>0.0093</td>
      <td>0.5110</td>
      <td>0.6729</td>
      <td>18.03%</td>
      <td>1.0005</td>
      <td>11.48%</td>
      <td>29.49%</td>
      <td>133 days</td>
    </tr>
    <tr>
      <th>LightGBM(target_20sharpe) Long positionDaily PnL(W)-price_weighted</th>
      <td>366</td>
      <td>0.03</td>
      <td>0.0000</td>
      <td>0.0006</td>
      <td>0.7745</td>
      <td>0.0141</td>
      <td>0.0001</td>
      <td>0.0106</td>
      <td>0.4901</td>
      <td>0.6507</td>
      <td>18.84%</td>
      <td>1.0005</td>
      <td>11.99%</td>
      <td>36.36%</td>
      <td>85 days</td>
    </tr>
    <tr>
      <th>XGB(target_20sharpe) Long positionDaily PnL(W)-cap_weighted</th>
      <td>366</td>
      <td>0.03</td>
      <td>0.0000</td>
      <td>0.0005</td>
      <td>0.7904</td>
      <td>0.0129</td>
      <td>0.0001</td>
      <td>0.0093</td>
      <td>0.4898</td>
      <td>0.6783</td>
      <td>17.88%</td>
      <td>1.0004</td>
      <td>11.39%</td>
      <td>28.46%</td>
      <td>88 days</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
