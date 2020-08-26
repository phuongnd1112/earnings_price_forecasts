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
#DataFrame displaying max R2 for each single feature
max_score = pd.DataFrame() #create DataFrame
ticker_sample = []  #empty lists to store variables 
feature_sample = [] 
score_sample = []

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
    fa_df = df = tcbs_ticker.ratio(ticker, period_type=0, period=40)  #call data from database
    fa_df = fa_df.rename(columns = {'YearReport': 'Year', 'LengthReport':'Quarter'}) #changing column names for convenience 
    fa_df = fa_df.sort_values(['Year', 'Quarter'], ascending = True) #sort time for better visualisation and fitting into income statement data

    fa_list = ['revenue', 'operationProfit', 'netProfit', 'provision', 'creditGrowth', 'cash', 'liability', 'equity', 'asset', 'customerCredit', 'priceToEarning', 'priceToBook', 'roe', 'bookValuePerShare', 'earningPerShare', 'profitMargin', 'provisionOnBadDebt', 'badDebtPercentage', 'loanOnDeposit', 'nonInterestOnToi'] #creating a list for the FA numerics we want to examine 

    for item in fa_list: 
        result[item] = (fa_df[item] - fa_df[item].shift(4)) / fa_df[item].shift(4) *100 #finding the YoY differences between each item of FA 

    result.dropna(inplace=True) 
    print(result.head())
    
    # ------ TRAINING MODEL 
    features_list = ['revenue', 'priceToEarning', 'priceToBook', 'roe', 'earningPerShare'] 
    print(features_list.index('priceToEarning'))
    for f in features_list: 
        current_index = features_list.index(f) 
        next_index = current_index + 1 
        append_list = features_list[next_index:]
        if not append_list: 
            pass
        else:
            for f2 in append_list: 
                features = result[[f, f2]]
                print(features)

                outcomes = result[['Ri (Max/Min)']]
                x_train, x_test, y_train, y_test = train_test_split(features, outcomes, train_size = 0.8, shuffle=False)  

                model = LinearRegression() 
                model.fit(x_train, y_train) 
                coefs = model.coef_ 
                intercepts = model.intercept_
                print(coefs) 
                print(intercepts)
                score = model.score(x_train, y_train) 
                score_test = model.score(x_test, y_test)
                predictions = model.predict(x_train)
                
                xx_pred, yy_pred = np.meshgrid(x_train[f], x_train[f2]) 
                model_vis = np.array([xx_pred.flatten(), yy_pred.flatten()]).T
                predictions = model.predict(model_vis)
                
                print(ticker) 
                print(score) 

                sns.set() 
                fig = plt.figure(figsize=[10,10])
                ax = fig.add_subplot(projection='3d') 
                ax.scatter(x_train[[f]], x_train[[f2]], y_train, color='forestgreen', alpha = 0.8)  
                ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predictions, facecolor='red', s=30, edgecolor='red', alpha=0.3) 
                ax.set_xlabel(str(f) + '\n' + 'y = ' + str(round(coefs[0][0], 5))+'*x' + ' + ' + str(round(intercepts[0],3))) 
                ax.set_ylabel(str(f2) + '\n' + 'y = ' + str(round(coefs[0][1], 3))+'*x' + ' + ' + str(round(intercepts[0],3)))
                ax.set_zlabel('Maximum Return Potential in %')
                ax.set_title('Correlation between Maximum Potential Returns, ' + str(f) + ' and ' + str(f2) + '. \n R-Squared: = ' + str(score))
                plt.savefig(local+'/'+str(f)+str(f2)+'ticker.png') 

                ticker_list = []
                feature_list = [] 
                score_list = []

                total = pd.DataFrame() 
                ticker_list.append(ticker) 
                total['ticker'] = ticker_list 
                print(ticker) 
                feature = f+f2 
                feature_list.append(feature) 
                total['feature'] = feature_list 
                score_list.append(score) 
                total['score'] = score_list
                print(score)

                mx_score = total['score'].max() 
                list_result = total.loc[total['score'] == mx_score].values.tolist()
                ticker_sample.append(list_result[0][0])
                feature_sample.append(list_result[0][1])
                score_sample.append(list_result[0][2])   

analyse_aggregate_feature('VCB') 
max_score['ticker'] = ticker_sample #indicating list & column location 
max_score['feature'] = feature_sample 
max_score['score'] = score_sample 
max_score.to_csv(local+'/file_aggregate.csv')