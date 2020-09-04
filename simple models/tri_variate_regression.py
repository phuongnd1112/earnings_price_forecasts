#TCBS DB specific - API needed to start 

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

# ----- PRELIM SET UP 
local = input('Path to save figures? ')
#user_ticker = input('Ticker: ')
#DataFrame displaying max R2 for each single feature
max_score = pd.DataFrame() #create DataFrame
ticker_sample = []  #empty lists to store variables 
feature_sample = [] 
score_sample = []

# FUNCTION : LEAST-SQUARED LINEAR MODEL FOR MULTIPLE FEATURES (DOCUMENTATION SAME AS ABOVE, EXCEPT FOR WHEN TRAINING MODEL) 
def analyse_aggregate_feature(ticker): 
    # ------ CLEANING AND PROCESSING DATA
    df = tcbs_market.stock_prices([ticker], period=2000) #pulling data, according to ticker 
    df = df.rename(columns = {'openPriceAdjusted': 'Open', 'closePriceAdjusted':'Price'}) #renaming column for familiriarity purposes 
    df['dateReport'] = pd.to_datetime(df['dateReport']) #ensuring that all date data is in time series 

    df.reset_index(inplace = True) #reset index 

    df['year'] = pd.DatetimeIndex(df['dateReport']).year #inputing year column - sorting purposes 
    df['quarter'] = pd.DatetimeIndex(df['dateReport']).quarter #inputing quarter column - sorting purposes

    df = df.sort_values('dateReport', ascending=True) #sort values - earlier first (like stocks data) 

    years = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020] #year list for iteration
    quarter = [1,2,3,4] #quarter list for iteration 

    #create lists to store processed data for new pd 
    year_list = [] 
    quarter_list = []
    r_list = []
    x_positions = [] 

    #looping through year/quarter list to find quarterly max and min
    for y in years: 
        df_1 = df.loc[df['year'] == y, :] #cut df by year 
        for q in quarter: 
            df_2 = df_1.loc[df['quarter'] == q, :] #cut df by quarter 
            if df_2.empty: 
                pass 
            else: 
                x_pos = df_2['Price'].argmax() - df_2['Price'].argmin() #finding delta x of maxima and minima 
                x_positions.append(1 if x_pos > 0 else -1)#signalling positions with -1 or 1 (1 for when maxima comes after minima and vv)
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
                

    result = pd.DataFrame() #storing new result in this dataFrame
    result['Year'] = year_list 
    result['Quarter'] = quarter_list 
    result['Ri (Max/Min)'] = r_list
    result['ArgMax/Min Position'] = x_positions
    result = result.sort_values(['Year', 'Quarter'], ascending = True) 

    # ------ FA Analysis 
    fa_df = df = tcbs_ticker.ratio(ticker, period_type=0, period=40)  #call financials data from database
    fa_df = fa_df.rename(columns = {'YearReport': 'Year', 'LengthReport':'Quarter'}) #changing column names for convenience 
    fa_df = fa_df.sort_values(['Year', 'Quarter'], ascending = True) #sort time for better visualisation and fitting into income statement data

    fa_list = ['revenue', 'operationProfit', 'netProfit', 'provision', 'creditGrowth', 'cash', 'liability', 'equity', 'asset', 'customerCredit', 'priceToEarning', 'priceToBook', 'roe', 'bookValuePerShare', 'earningPerShare', 'profitMargin', 'provisionOnBadDebt', 'badDebtPercentage', 'loanOnDeposit', 'nonInterestOnToi'] #this list has all column names from fa_df.columns.to_list() 

    for item in fa_list: #appending to result df for processing purposes 
        result[item] = (fa_df[item] - fa_df[item].shift(4)) / fa_df[item].shift(4) *100 #finding the YoY differences between each item of FA 

    result.dropna(inplace=True) #avoid error 
    print(result.head())
    
    # ------ TRAINING MODEL - MULTIVARIATE REGRESSION MODEL TO DETERMINE BEST DETERMINANTS FOR MAXIMUM POTENTIAL CHANGES 
    features_list = ['revenue', 'priceToEarning', 'priceToBook', 'roe', 'earningPerShare'] #taken from fa_list[], slice and append to as many needed 
    for f in features_list: #looping through the entire list, item by item - loop 1
        current_index = features_list.index(f) #find the current item index 
        next_index = current_index + 1 #assigning value for proceding indeces 
        append_list = features_list[next_index:] #create a new list for each starting item (as they will only iterate through items after them)
        if not append_list: #this returns one empty list, so pass empty to avoid error
            pass
        else:
            for f2 in append_list: #looping through the append list (in reality, just features_list but without the items in the current and preceding indeces) 
                features = result[[f, f2]] #setting feature bins in 2 
                outcomes = result[['Ri (Max/Min)']] #outcomes = maximum potential change 

                x_train, x_test, y_train, y_test = train_test_split(features, outcomes, train_size = 0.8, shuffle=False)  #splitting train/test dataset for better estimation / note, shuffle is False because we want to rely on time seried historical data

                model = LinearRegression() #fitting model 
                model.fit(x_train, y_train) #fit train x and y 
                coefs = model.coef_ #calculate coefficient - multiplicative factor of each feature to outcome
                intercepts = model.intercept_ #calculate intercept 
                #coefficient and intercept will give formula to numerically calculate model 
                score = model.score(x_train, y_train) #score train model 
                score_test = model.score(x_test, y_test) #score test model 
                predictions = model.predict(x_train) #pass x-train through predict to determine best fit plane 
                
                xx_pred, yy_pred = np.meshgrid(x_train[f], x_train[f2]) #creating meshgrid for best-fit plane
                model_vis = np.array([xx_pred.flatten(), yy_pred.flatten()]).T #reshaping array 
                predictions = model.predict(model_vis) #passing reshaped array through predict to fit plane
                
                print(ticker) 
                print(score) 

                sns.set() 
                fig = plt.figure(figsize=[10,10])
                ax = fig.add_subplot(projection='3d') #allowing for 3d model 
                ax.scatter(x_train[[f]], x_train[[f2]], y_train, color='forestgreen', alpha = 0.8) #scattering train data 
                ax.scatter(x_test[[f]], x_test[[f2]], y_test, color='magenta', alpha = 0.5) #scattering test data
                ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predictions, facecolor='red', s=30, edgecolor='red', alpha=0.3) #fit plane 
                ax.set_xlabel(str(f) + '\n' + 'y = ' + str(round(coefs[0][0], 5))+'*x' + ' + ' + str(round(intercepts[0],3))) #set x label with regression formula 
                ax.set_ylabel(str(f2) + '\n' + 'y = ' + str(round(coefs[0][1], 3))+'*x' + ' + ' + str(round(intercepts[0],3))) #set y label with regression formula 
                ax.set_zlabel('Maximum Return Potential in %') #set z label 
                ax.set_title('Correlation between Maximum Potential Returns, ' + str(f) + ' and ' + str(f2) + '. \n R-Squared: = ' + str(score)) #set title 
                plt.savefig(local+'/'+str(f)+str(f2)+ticker+'.png') #save fig 

                #lists to numerically store variables 
                ticker_list = []
                feature_list = [] 
                score_list = []

                total = pd.DataFrame() #dataFrame to store ticker, feature combos and R2 
                ticker_list.append(ticker) 
                total['ticker'] = ticker_list 
                print(ticker) 
                feature = f+f2 #creating feature combos 
                feature_list.append(feature) 
                total['feature'] = feature_list 
                score_list.append(score) #finding score 
                total['score'] = score_list
                print(score)

                mx_score = total['score'].max() #max score gets appending 
                list_result = total.loc[total['score'] == mx_score].values.tolist()
                #because data was pulled from dataFrame, use indeces to locate values 
                ticker_sample.append(list_result[0][0]) 
                feature_sample.append(list_result[0][1])
                score_sample.append(list_result[0][2])   

ticker_list = ['ACB','TPB', 'VCB'] #create list with tickers to examine 
for t in ticker_list: #calling each function of each ticker, by looking through ticker list 
    analyse_aggregate_feature(t) 

max_score['ticker'] = ticker_sample #indicating list & column location 
max_score['feature'] = feature_sample 
max_score['score'] = score_sample 
max_score.to_csv(local+'/file_aggregate.csv') #save table 
