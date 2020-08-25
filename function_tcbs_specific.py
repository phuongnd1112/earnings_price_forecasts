# ------ SET UP 
#pip3 install pandas numpy seaborn matplotlib sklearn 
#pip install pandas numpy seaborn matplotlib sklearn 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import tcdata.stock.llv.finance as tcbs
import tcdata.stock.llv.market as tcbs_market 
import tcdata.stock.llv.ticker as tcbs_ticker 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#DataFrame displaying max R2 for each single feature
max_score = pd.DataFrame() #create DataFrame
ticker_sample = []  #empty lists to store variables 
feature_sample = [] 
score_sample = []

# FUNCTION 1: LEAST-SQUARED LINEAR MODEL FOR SINGLE FEATURE 
def analyse_single_feature(ticker): 
    # ------ CLEANING AND PROCESSING DATA
    df = tcbs_market.stock_prices([ticker], period=2000) #getting data from tcbs_market databse / replaceable with data from yfinance
    df = df.rename(columns = {'openPriceAdjusted': 'Open', 'closePriceAdjusted':'Price'}) #change column name for convenience
    df['dateReport'] = pd.to_datetime(df['dateReport']) #setting timeseries 

    df.reset_index(inplace = True) #index reset 

    df['year'] = pd.DatetimeIndex(df['dateReport']).year #create year column to sort 
    df['quarter'] = pd.DatetimeIndex(df['dateReport']).quarter #create quarter column to sort 

    df = df.sort_values('dateReport', ascending=True) #sort column by ascending to fit traditional stocks chart 

    years = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020] #create list of years 
    quarter = [1,2,3,4] #create list of quarters 
    
    #empty list to store data after processing 
    year_list = [] #year 
    quarter_list = [] #quarter 
    r_list = [] #values of R(i) 
    x_positions = [] #x_pos whether max comes before min and vv. 

    #slicing data accordingly (splitting data by year/quarter) --> calculating quartely maximum potential
    for y in years: #looping through the year list 
        df_1 = df.loc[df['year'] == y, :] #slicing data by year 
        for q in quarter: #looping through the quarter list 
            df_2 = df_1.loc[df['quarter'] == q, :] #slicing data (alr sliced by year) by quarter 
            if df_2.empty: #some DF might be empty, so print 0 instead of further processing 
                pass
            else: #for dataframe with data 
                x_pos = df_2['Price'].argmax() - df_2['Price'].argmin() #finding delta x of maxima and minima 
                x_positions.append(1 if x_pos > 0 else -1) #signalling positions with -1 or 1 (1 for when maxima comes after minima and vv)
                if x_pos > 0: #if maxima comes after minima, we want to find max/min-1 
                    r_i = df_2['Price'].max() / df_2['Price'].min() - 1
                    r_list.append(r_i)
                    year_list.append(y) 
                    quarter_list.append(q)
                else: #if minima comes after maxima, we want to find min/max-1 (this will signal a negative downturn) 
                    r_i = df_2['Price'].min() / df_2['Price'].max() - 1
                    r_list.append(r_i) 
                    year_list.append(y) 
                    quarter_list.append(q)
                
    result = pd.DataFrame() #creating empty df to store results 
    #append results from above loop 
    result['Year'] = year_list 
    result['Quarter'] = quarter_list 
    result['Ri (Max/Min)'] = r_list
    result['ArgMax/Min Position'] = x_positions
    result = result.sort_values(['Year', 'Quarter'], ascending = True) #sort time for better visualisation and fitting into income statement data 

    # ------ IMPORT FA DATA  
    fa_df = tcbs.finance_income_statement(ticker, period_type = 0, period = 40) #call data from database
    fa_df = fa_df.rename(columns = {'YearReport': 'Year', 'LengthReport':'Quarter'}) #changing column names for convenience 
    fa_df = fa_df.sort_values(['Year', 'Quarter'], ascending = True) #sort time for better visualisation and fitting into income statement data

    
    fa_list = ['Total_Operating_Income', 'Net_Profit'] #creating a list for the FA numerics we want to examine 

    for item in fa_list: 
        result[item] = (fa_df[item] - fa_df[item].shift(4)) / fa_df[item].shift(4) *100 #finding the YoY differences between each item of FA 

    result = result.dropna() 
    #print(result)
    
    # ------ TRAINING MODEL 
    features_list = ['Net_Profit', 'Total_Operating_Income'] #because we are training single feature models, create features list 

    feature_list = [] 
    score_list = [] 
    ticker_list = []

    for f in features_list: #looping through each feature 
        features = result[[f]] 
        outcomes = result[['Ri (Max/Min)']]

        x_train, x_test, y_train, y_test = train_test_split(features, outcomes, train_size = 0.9) #dividing test/train bins 

        model = LinearRegression() #create model 
        model.fit(x_train, y_train) #fit model 
        score = model.score(x_train, y_train) #score train data
        score_test = model.score(x_test, y_test) #score test data 

        total = pd.DataFrame() 
        ticker_list.append(ticker) 
        total['ticker'] = ticker_list 
        print(ticker) 
        feature_list.append(f) 
        total['feature'] = feature_list
        print(f)
        score_list.append(score) 
        total['score'] = score_list
        print(score)

    mx_score = total['score'].max() 
    list_result = total.loc[total['score'] == mx_score].values.tolist()
    print(list_result)
    ticker_sample.append(list_result[0][0])
    feature_sample.append(list_result[0][1])
    score_sample.append(list_result[0][2])
    #print(dataframe) 

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

    x_train, x_test, y_train, y_test = train_test_split(features, outcomes, train_size = 0.7) 

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

      
#HOW TO CALL FUNCTION 
ticker_list = ['ACB', 'TCB', 'TPB', 'VCB', 'BID', 'VPB', 'VIB', 'MBB', 'CTG', 'SHB', 'STB', 'HDB', 'LPB'] #create list with tickers to examine 
for t in ticker_list: #calling each function of each ticker, by looking through ticker list 
    analyse_aggregate_feature(t) 
    analyse_single_feature(t) 

max_score['ticker'] = ticker_sample #indicating list & column location 
max_score['feature'] = feature_sample 
max_score['score'] = score_sample 
print(max_score)
max_score.to_csv('/Users/phuongd/Desktop/file.csv') 