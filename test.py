#importing needed libraries 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import tcdata.stock.llv.finance as tcbs
import tcdata.stock.llv.market as tcbs_market 
import tcdata.stock.llv.ticker as tcbs_ticker 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

ticker_list = ['ACB', 'TCB', 'TPB', 'VCB', 'BID', 'VPB', 'VIB']

def analyse_single_feature(ticker): 
    df = tcbs_market.stock_prices([ticker], period=2000) 
    df = df.rename(columns = {'openPriceAdjusted': 'Open', 'closePriceAdjusted':'Price'}) #change column name for convenience
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
                print(0)
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
    fa_df = tcbs.finance_incomestatement(ticker, periodtype = 'Q', period = 40)
    fa_df = fa_df.rename(columns = {'YearReport': 'Year', 'LengthReport':'Quarter'})
    fa_df = fa_df.sort_values(['Year', 'Quarter'], ascending = True) 


    fa_list = ['Total_Operating_Income', 'Net_Profit']

    for item in fa_list: 
        result[item] = (fa_df[item].shift(0) - fa_df[item].shift(4)) / fa_df[item].shift(4) *100 

    #Train Model 
    features_list = ['Net_Profit', 'Total_Operating_Income']
    for f in features_list: 
        features = result[[f]] 
        outcomes = result[['Ri (Max/Min)']]

        x_train, x_test, y_train, y_test = train_test_split(outcomes, features, train_size = 0.7) 

        model = LinearRegression() 
        model.fit(x_train, y_train) 
        score = model.score(x_train, y_train) 
        score_test = model.score(x_test, y_test)

        print(ticker) 
        print(score) 
        print(f)

def analyse_aggregate_feature(ticker): 
    df = tcbs_market.stock_prices([ticker], period=2000) 
    df = df.rename(columns = {'openPriceAdjusted': 'Open', 'closePriceAdjusted':'Price'}) #change column name for convenience
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
                print(0)
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
    fa_df = tcbs.finance_incomestatement(ticker, periodtype = 'Q', period = 40)
    fa_df = fa_df.rename(columns = {'YearReport': 'Year', 'LengthReport':'Quarter'})
    fa_df = fa_df.sort_values(['Year', 'Quarter'], ascending = True) 


    fa_list = ['Total_Operating_Income', 'Net_Profit']

    for item in fa_list: 
        result[item] = (fa_df[item].shift(0) - fa_df[item].shift(4)) / fa_df[item].shift(4) *100 

    #Train Model 
    features = result[['Net_Profit', 'Total_Operating_Income']] 
    outcomes = result[['Ri (Max/Min)']]

    x_train, x_test, y_train, y_test = train_test_split(outcomes, features, train_size = 0.7) 

    model = LinearRegression() 
    model.fit(x_train, y_train) 
    score = model.score(x_train, y_train) 
    score_test = model.score(x_test, y_test)

    print(ticker) 
    print(score) 
    print(f)
      
#for t in ticker_list: 
    analyse_aggregate_feature(t) 
    analyse_single_feature(t)