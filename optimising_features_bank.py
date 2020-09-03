#IMPORT LIBRARIES 
#pip3 install pandas numpy seaborn matplotlib sklearn
#pip install pandas numpy seaborn matplotlib sklearn 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import tcdata.stock.llv.finance as tcbs_fin
import tcdata.stock.llv.market as tcbs_market 
import tcdata.stock.llv.ticker as tcbs_ticker 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from itertools import chain, combinations

#PART 1 - FETCHING USER INFO AND STRUCTURING 
user_path = input('Save figures and data to: ') #path to save fig
user_ticker = input('Ticker: ') #takes in ticker 
user_mode = input('Simple or Aggregate Model (simple/aggregate)?: ') #choosing between simple and aggregate models (see documentation)
features_pool = input('Which list would you list to run? (ratios/income/balance): ') #choosing between features list (see documentation) 
user_mean = input('Mean Option: ') #choosing between 2 returns indicators (see documentation)
max_score = pd.DataFrame() #create DataFrame to store master result 
ticker_sample = []  #empty list to store ticker variables 
feature_sample = [] #empty list to store features variables 
score_sample = [] #empty list to score R2 scores 

def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset
    dataset = dataset 

class optimising_features_score(): 
    def clean_data(self): 
        df_dict = tcbs_market.stock_prices([user_ticker], period=2000) #gettinf data 
        self.df = df_dict[user_ticker] 
        self.df = self.df.rename(columns = {'OpenPrice_Adjusted': 'Open', 'ClosePrice_Adjusted':'Close'}) #change column name for convenience

        self.df['year'] = pd.DatetimeIndex(self.df.index).year #create year column to sort 
        self.df['quarter'] = pd.DatetimeIndex(self.df.index).quarter
    
        self.df = self.df.sort_index(ascending=True) 
        print(self.df) 
    
    def generate_quarter_data (self): 
        self.years = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
        self.quarters = [1,2,3,4] 

        year_list = [] 
        quarter_list = [] 
        r_list = [] 
        qmean_list = [] 
        x_positions = [] 

        for y in self.years: #looping through the year list 
            df_1 = self.df.loc[self.df['year'] == y, :] #slicing data by year 
            for q in self.quarters: #looping through the quarter list 
                df_2 = df_1.loc[self.df['quarter'] == q, :] #slicing data (alr sliced by year) by quarter 
                if df_2.empty: #some DF might be empty, so print 0 instead of further processing 
                    pass
                else: #for dataframe with data 
                    x_pos = df_2['Close'].argmax() - df_2['Close'].argmin() #finding delta x of maxima and minima 
                    x_positions.append(1 if x_pos > 0 else -1) #signalling positions with -1 or 1 (1 for when maxima comes after minima and vv)
                    if x_pos > 0: #if maxima comes after minima, we want to find max/min-1 
                        r_i = df_2['Close'].max() / df_2['Close'].min() - 1
                        q_mean = df_2['Close'].mean() 
                        r_list.append(r_i)
                        year_list.append(y) 
                        quarter_list.append(q)
                        qmean_list.append(q_mean) 
                    else: #if minima comes after maxima, we want to find min/max-1 (this will signal a negative downturn) 
                        r_i = df_2['Close'].min() / df_2['Close'].max() - 1
                        q_mean = df_2['Close'].mean() 
                        r_list.append(r_i) 
                        year_list.append(y) 
                        quarter_list.append(q)
                        qmean_list.append(q_mean) 
        
        self.result = pd.DataFrame() #creating empty df to store results 
        #append results from above loop 
        self.result['Year'] = year_list 
        self.result['Quarter'] = quarter_list 
        self.result['Q_Mean'] =  qmean_list 
        self.result['Rq'] = (self.result['Q_Mean'] - self.result['Q_Mean'].shift(1)) / self.result['Q_Mean'].shift(1)
        self.result['Ri'] = r_list
        self.result['ArgMax/Min Position'] = x_positions
        self.result = self.result.sort_values(['Year', 'Quarter'], ascending = True) #sort time for better visualisation and fitting into income statement data 
        self.total_period = len(self.result.index) 
        self.result = self.result.fillna(0)        
        print(self.result) 

    def import_financials(self): 
        self.ratio_df = tcbs_ticker.ratio(user_ticker, period_type=0, period=self.total_period) 
        self.ratio_df = self.ratio_df.sort_values(['YearReport', 'LengthReport'], ascending=True)  
        self.ratios_list = ['revenue', 'operationProfit', 'netProfit', 'provision', 'creditGrowth', 'cash', 'liability', 'equity', 'asset', 'customerCredit', 'noStock', 'capitalize', 'priceToEarning', 'priceToBook', 'roe', 'bookValuePerShare', 'earningPerShare', 'profitMargin', 'provisionOnBadDebt', 'badDebtPercentage', 'loanOnDeposit', 'nonInterestOnToi', 'roa', 'betaIndex', 'dividend'] 

        self.ratios_returns = pd.DataFrame()
        self.ratios_returns['YearReport'] = self.ratio_df['YearReport']
        self.ratios_returns['LengthReport'] = self.ratio_df['LengthReport']
        for item in self.ratios_list: 
            self.ratios_returns[item+'_YoYReturn'] = (self.ratio_df[item] - self.ratio_df[item].shift(-1)) / self.ratio_df[item].shift(-1) 
        self.ratios_returns = self.ratios_returns.fillna(0) 
        correlation(self.ratios_returns, 0.8)
        print(self.ratios_returns)
        self.ratios = self.ratios_returns.columns.tolist()

        self.balancesheet_df = tcbs_fin.finance_balance_sheet(user_ticker, period_type=0, period=self.total_period)
        self.balancesheet_df = self.balancesheet_df.sort_values(['YearReport', 'LengthReport'], ascending=True)
        self.balance_list = ['cash', 'liability', 'fixedAsset', 'equity', 'asset', 'customerCredit', 'centralBankDeposit', 'otherBankDeposit', 'otherBankLoan', 'StockInvest', 'payableInterest', 'customerLoan', 'netCustomerLoan', 'provision', 'otherAsset', 'oweCentralBank', 'oweOtherBank', 'otherBankCredit', 'pricePaper', 'otherPayable', 'fund', 'undistributedIncome', 'minorShareHolderProfit', 'capital', 'receivableInterest', 'YearReport_BK', 'LengthReport_BK', 'badDebt']

        self.balance_returns = pd.DataFrame()
        self.balance_returns['YearReport'] = self.balancesheet_df['YearReport']
        self.balance_returns['LengthReport'] = self.balancesheet_df['LengthReport']
        for item in self.balance_list: 
            self.balance_returns[str(item)+'_YoYReturn'] = (self.balancesheet_df[item] - self.balancesheet_df[item].shift(-1)) / self.balancesheet_df[item].shift(-1) 
        self.balance_returns = self.balance_returns.fillna(0) 
        correlation(self.balance_returns, 0.8) 
        print(self.balance_returns)
        self.balance = self.balance_returns.columns.tolist()

        self.income_df = tcbs_fin.finance_income_statement(user_ticker, period_type=0, period=self.total_period) 
        self.income_df = self.income_df.sort_values(['YearReport', 'LengthReport'], ascending=True) 
        self.income_list = ['Net_Interest_Income', 'Net_Fee_Income', 'Net_Investment_Income', 'Other_Income', 'Total_Operating_Income', 'Operating_Expenses', 'Pre_Provision_Income', 'Provision_Expenses', 'Net_Profit_Before_Tax', 'Net_Profit_After_Tax', 'Net_Profit']

        self.income_returns = pd.DataFrame()
        self.income_returns['YearReport'] = self.income_df['YearReport']
        self.income_returns['LengthReport'] = self.income_df['LengthReport']
        for item in self.income_list: 
            self.income_returns[str(item)+'_YoYReturn'] = (self.income_df[item] - self.income_df[item].shift(-1)) / self.income_df[item].shift(-1)
        self.income_returns = self.income_returns.fillna(0) 
        correlation(self.income_returns, 0.8) 
        print(self.income_returns)
        self.income = self.income_returns.columns.tolist()

    def single_feature(self, feauture_count, mean_option): 
        if features_pool == 'ratios': 
            features_list = self.ratios[2:]
            feature_list = [] 
            score_list = [] 
            ticker_list = []

            for f in features_list: 
                features = self.ratios_returns[[f]]
                
                outcome = self.result[[mean_option]]

                x_train, x_test, y_train, y_test = train_test_split(features, outcome, train_size = 0.8, shuffle=False)

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
                plt.title('R^2 of ' + f + ' for ' + user_ticker + "\n" + 'y = ' + str(round(coefs[0][0], 5))+'*x' + ' + ' + str(round(intercepts[0],3)))
                plt.xlabel('% Δ ' + f + ' YoY')
                plt.ylabel('Maximum Stocks Potential/Quarter')
                plt.savefig(user_path+'/'+features_pool+f+'.png')  

                total = pd.DataFrame() 
                ticker_list.append(user_ticker) 
                total['ticker'] = ticker_list 
                print(user_ticker) 
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

            max_score['ticker'] = ticker_sample #indicating list & column location 
            max_score['feature'] = feature_sample 
            max_score['score'] = score_sample 
        
        if features_pool == 'balance': 
            features_list = self.balance[2:]

            feature_list = [] 
            score_list = [] 
            ticker_list = []

            for f in features_list: 
                features = self.balance_returns[[f]]
                
                outcome = self.result[[mean_option]]

                x_train, x_test, y_train, y_test = train_test_split(features, outcome, train_size = 0.8, shuffle=False)

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
                plt.title('R^2 of ' + f + ' for ' + user_ticker + "\n" + 'y = ' + str(round(coefs[0][0], 5))+'*x' + ' + ' + str(round(intercepts[0],3)))
                plt.xlabel('% Δ ' + f + ' YoY')
                plt.ylabel('Maximum Stocks Potential/Quarter')
                plt.savefig(user_path+'/'+features_pool+f+'.png')  

                total = pd.DataFrame() 
                ticker_list.append(user_ticker) 
                total['ticker'] = ticker_list 
                print(user_ticker) 
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

            max_score['ticker'] = ticker_sample #indicating list & column location 
            max_score['feature'] = feature_sample 
            max_score['score'] = score_sample 
        
        if features_pool == 'income': 
            features_list = self.income[2:]

            feature_list = [] 
            score_list = [] 
            ticker_list = []

            for f in features_list: 
                features = self.income_returns[[f]]
                
                outcome = self.result[[mean_option]]

                x_train, x_test, y_train, y_test = train_test_split(features, outcome, train_size = 0.8, shuffle=False)

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
                plt.title('R^2 of ' + f + ' for ' + user_ticker + "\n" + 'y = ' + str(round(coefs[0][0], 5))+'*x' + ' + ' + str(round(intercepts[0],3)))
                plt.xlabel('% Δ ' + f + ' YoY')
                plt.ylabel('Maximum Stocks Potential/Quarter')
                plt.savefig(user_path+'/'+features_pool+f+'.png') 

                total = pd.DataFrame() 
                ticker_list.append(user_ticker) 
                total['ticker'] = ticker_list 
                print(user_ticker) 
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

    def multi_features(self, features_pool, mean_option): 
        def powerset(iterable):
            select = list(iterable)  # allows duplicate elements
            return chain.from_iterable(combinations(select, r) for r in range(len(select)+1))

        if features_pool == 'ratios': 
            ratio_pool = self.ratios[2:]

            combo_list = [] 

            for combo in enumerate(powerset(ratio_pool), 1):
                f = combo
                combo_list.append(f[1])

            feature_list = [] 
            score_list = [] 
            ticker_list = []
            coefs_list = []
            intercepts_list = []

            for c in combo_list[1:]: 
                x = list(c) 
                features = self.ratios_returns[x]
                outcome = self.result[[mean_option]]
        
                x_train, x_test, y_train, y_test = train_test_split(features, outcome, train_size = 0.8, shuffle=False)

                model = LinearRegression() #create model 
                model.fit(x_train, y_train) #fit model 
                score = model.score(x_train, y_train) #score train model 
                score_test = model.score(x_test, y_test) #score test model 
                predictions = model.predict(x_train) #pass x-train through predict to determine best fit plane 
                coefs = model.coef_ #calculate coefficient - multiplicative factor of each feature to outcome
                intercepts = model.intercept_ #calculate intercept
                print(coefs, intercepts) 

                total = pd.DataFrame() 
                ticker_list.append(user_ticker) 
                total['ticker'] = ticker_list 
                print(user_ticker) 
                feature_list.append(c) 
                total['feature'] = feature_list
                print(c)
                score_list.append(score) 
                total['score'] = score_list
                print(score)
                coefs_list.append(coefs) 
                total['coef'] = coefs_list 
                intercepts_list.append(intercepts) 
                total['intercept'] = intercepts_list
                total.to_csv(user_path+'/ratio_agg.csv')
        
        if features_pool == 'balance': 
            balance_pool = self.balance[2:]

            combo_list = [] 

            for combo in enumerate(powerset(balance_pool), 1):
                f = combo
                combo_list.append(f[1])

            feature_list = [] 
            score_list = [] 
            ticker_list = []
            coefs_list = [] 
            intercepts_list =[]

            for c in combo_list[1:]: 
                print(c) 
                features = self.balance_returns[list(c)]
                outcome = self.result[[mean_option]]
        
                x_train, x_test, y_train, y_test = train_test_split(features, outcome, train_size = 0.8, shuffle=False)

                model = LinearRegression() #create model 
                model.fit(x_train, y_train) #fit model 
                score = model.score(x_train, y_train) #score train model 
                score_test = model.score(x_test, y_test) #score test model 
                predictions = model.predict(x_train) #pass x-train through predict to determine best fit plane 
                coefs = model.coef_ #calculate coefficient - multiplicative factor of each feature to outcome
                intercepts = model.intercept_ #calculate intercept

                total = pd.DataFrame() 
                ticker_list.append(user_ticker) 
                total['ticker'] = ticker_list 
                print(user_ticker) 
                feature_list.append(c) 
                total['feature'] = feature_list
                print(c)
                score_list.append(score) 
                total['score'] = score_list
                print(score)
                coefs_list.append(coefs) 
                total['coef'] = coefs_list 
                intercepts_list.append(intercepts) 
                total['intercept'] = intercepts_list
                total.to_csv(user_path+'/balance_agg.csv')

        if features_pool == 'income': #DONE
            income_pool = self.income[2:]

            combo_list = [] 

            for combo in enumerate(powerset(income_pool), 1):
                f = combo
                combo_list.append(f[1])

            feature_list = [] 
            score_list = [] 
            ticker_list = []
            coefs_list = [] 
            intercepts_list =[]

            for c in combo_list[1:]: 
                print(c) 
                features = self.income_returns[list(c)]
                outcome = self.result[[mean_option]]
        
                x_train, x_test, y_train, y_test = train_test_split(features, outcome, train_size = 0.8, shuffle=False)

                model = LinearRegression() #create model 
                model.fit(x_train, y_train) #fit model 
                score = model.score(x_train, y_train) #score train model 
                score_test = model.score(x_test, y_test) #score test model 
                predictions = model.predict(x_train) #pass x-train through predict to determine best fit plane 
                coefs = model.coef_ #calculate coefficient - multiplicative factor of each feature to outcome
                intercepts = model.intercept_ #calculate intercept

                total = pd.DataFrame() 
                ticker_list.append(user_ticker) 
                total['ticker'] = ticker_list 
                print(user_ticker) 
                feature_list.append(c) 
                total['feature'] = feature_list
                print(c)
                score_list.append(score) 
                total['score'] = score_list
                print(score)
                coefs_list.append(coefs) 
                total['coef'] = coefs_list 
                intercepts_list.append(intercepts) 
                total['intercept'] = intercepts_list
                total.to_csv(user_path+'/income_agg.csv')

new = optimising_features_score() 
new.clean_data() 
new.generate_quarter_data() 
new.import_financials() 
if user_mode == 'simple': 
    new.single_feature(features_pool, user_mean)
    '''
    max_score['ticker'] = ticker_sample #indicating list & column location 
    max_score['feature'] = feature_sample 
    max_score['score'] = score_sample 
    max_score.to_csv(user_path+'/regression_table_single.csv')'''
elif user_mode == 'aggregate': 
    new.multi_features(features_pool, user_mean) 
else: 
    print("Error")