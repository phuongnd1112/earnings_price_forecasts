import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import tcdata.stock.llv.finance as tcbs
import tcdata.stock.llv.market as tcbs_market 
import tcdata.stock.llv.ticker as tcbs_ticker 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# FUNCTION 2: LEAST-SQUARED LINEAR MODEL FOR MULTIPLE FEATURES (DOCUMENTATION SAME AS ABOVE, EXCEPT FOR WHEN TRAINING MODEL) 
def analyse_aggregate_feature(ticker): 
    # ------ CLEANING AND PROCESSING DATA
    df = tcbs_market.stock_prices([ticker], period=2000) 
    df = df.rename(columns = {'openPriceAdjusted': 'Open', 'closePriceAdjusted':'Price'}) 
    df['dateReport'] = pd.to_datetime(df['dateReport'])

    df.reset_index(inplace = True)

    df['year'] = pd.DatetimeIndex(df['dateReport']).year
    df['quarter'] = pd.DatetimeIndex(df['dateReport']).quarter

    df = df.sort_values('dateReport', ascending=True) 

    years = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020] 
    quarter = [1,2,3,4] 

    year_list = [] 
    quarter_list = []
    r_list = []
    x_positions = [] 

    for y in years: 
        df_1 = df.loc[df['year'] == y, :] 
        for q in quarter: 
            df_2 = df_1.loc[df['quarter'] == q, :]
            if df_2.empty: 
                pass
            else: 
                x_pos = df_2['Price'].argmax() - df_2['Price'].argmin() 
                x_positions.append(1 if x_pos > 0 else -1)
                if x_pos > 0:
                    r_i = df_2['Price'].max() / df_2['Price'].min() - 1
                    r_list.append(r_i)
                    year_list.append(y) 
                    quarter_list.append(q)
                else:
                    r_i = df_2['Price'].min() / df_2['Price'].max() - 1
                    r_list.append(r_i) 
                    year_list.append(y) 
                    quarter_list.append(q)
                

    result = pd.DataFrame() 
    result['Year'] = year_list 
    result['Quarter'] = quarter_list 
    result['Ri (Max/Min)'] = r_list
    result['ArgMax/Min Position'] = x_positions
    result = result.sort_values(['Year', 'Quarter'], ascending = True) 

    # ------ FA Analysis 
    fa_df = tcbs.finance_income_statement(ticker, period_type = 0, period = 40)
    fa_df = fa_df.rename(columns = {'YearReport': 'Year', 'LengthReport':'Quarter'})
    fa_df = fa_df.sort_values(['Year', 'Quarter'], ascending = True) 


    fa_list = ['Total_Operating_Income', 'Net_Profit']

    for item in fa_list: 
        result[item] = (fa_df[item].shift(0) - fa_df[item].shift(4)) / fa_df[item].shift(4) *100 

    result = result.dropna() 
    #print(result) 

    # ------ TRAINING MODEL 
    features = result[['Net_Profit', 'Total_Operating_Income']] #except of looping through, aggregate features into list 
    outcomes = result[['Ri (Max/Min)']]

    x_train, x_test, y_train, y_test = train_test_split(features, outcomes, train_size = 0.8) 

    model = LinearRegression() 
    model.fit(x_train, y_train) 
    score = model.score(x_train, y_train) 
    score_test = model.score(x_test, y_test)
    
    print(ticker) 
    print(score) 

    ticker_list = [] 
    score_list = []

    total = pd.DataFrame() 
    ticker_list.append(ticker) 
    total['ticker'] = ticker_list 
    print(ticker) 
    score_list.append(score) 
    total['score'] = score_list
    print(score)

    print(total)