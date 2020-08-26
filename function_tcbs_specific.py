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

local = input('Path to save figures? ')
#user_ticker = [input('Ticker (if multiple, enter values separated by comma): ')]
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
    fa_df = df = tcbs_ticker.ratio(ticker, period_type=0, period=40)  #call data from database
    fa_df = fa_df.rename(columns = {'YearReport': 'Year', 'LengthReport':'Quarter'}) #changing column names for convenience 
    fa_df = fa_df.sort_values(['Year', 'Quarter'], ascending = True) #sort time for better visualisation and fitting into income statement data

    fa_list = ['revenue', 'operationProfit', 'netProfit', 'provision', 'creditGrowth', 'cash', 'liability', 'equity', 'asset', 'customerCredit', 'priceToEarning', 'priceToBook', 'roe', 'bookValuePerShare', 'earningPerShare', 'profitMargin', 'provisionOnBadDebt', 'badDebtPercentage', 'loanOnDeposit', 'nonInterestOnToi'] #creating a list for the FA numerics we want to examine 

    for item in fa_list: 
        result[item] = (fa_df[item] - fa_df[item].shift(4)) / fa_df[item].shift(4) *100 #finding the YoY differences between each item of FA 

    result.dropna(inplace=True) 
    print(result.head())
    
    # ------ TRAINING MODEL 
    features_list = ['revenue', 'operationProfit', 'netProfit', 'cash', 'liability', 'equity', 'asset', 'priceToEarning', 'priceToBook', 'roe', 'bookValuePerShare', 'earningPerShare', 'profitMargin', 'provisionOnBadDebt', 'badDebtPercentage', 'loanOnDeposit', 'nonInterestOnToi'] #because we are training single feature models, create features list 

    feature_list = [] 
    score_list = [] 
    ticker_list = []

    for f in features_list: #looping through each feature 
        features = result[[f]] 
        outcomes = result[['Ri (Max/Min)']]

        x_train, x_test, y_train, y_test = train_test_split(features, outcomes, train_size = 0.8, shuffle=False) #dividing test/train bins 

        model = LinearRegression() #create model 
        model.fit(x_train, y_train) #fit model 
        score = model.score(x_train, y_train) #score train model 
        score_test = model.score(x_test, y_test) #score test model 
        predictions = model.predict(x_train) #pass x-train through predict to determine best fit plane 
        coefs = model.coef_ #calculate coefficient - multiplicative factor of each feature to outcome
        intercepts = model.intercept_ #calculate intercept

        sns.set() 
        plt.figure(figsize = [10,10])
        plt.scatter(x_train, y_train, color='darkcyan', alpha=0.4) 
        plt.plot(x_train, predictions, color='darkorange')
        plt.title('R^2 of ' + f + ' for ' + ticker + "\n" + 'y = ' + str(round(coefs[0][0], 5))+'*x' + ' + ' + str(round(intercepts[0],3)))
        plt.xlabel('% Δ ' + f + ' YoY')
        plt.ylabel('Maximum Stocks Potential/Quarter')
        plt.savefig(local+"/"+f+ticker+'.png')

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

ticker_list = ['ACB','TPB', 'VCB', 'STB', 'LPB'] #create list with tickers to examine 
for t in ticker_list: #calling each function of each ticker, by looking through ticker list 
    analyse_single_feature(t) 
    
max_score['ticker'] = ticker_sample #indicating list & column location 
max_score['feature'] = feature_sample 
max_score['score'] = score_sample 
max_score.to_csv(local+'/file.csv') 

