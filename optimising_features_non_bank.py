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

#set up
#because we are considering a lot of variables that might have co-linearity, remove variables with this function 
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

#CLASS - optimise R2 in all features for ticker 
class optimising_features_score(): 
    # ---- PART 1: CLEAN DATA
    def clean_data(self): 
        df_dict = tcbs_market.stock_prices([user_ticker], period=2000) #getting data (tcbs api required)
        self.df = df_dict[user_ticker] #get item from dict 
        self.df = self.df.rename(columns = {'OpenPrice_Adjusted': 'Open', 'ClosePrice_Adjusted':'Close'}) #change column name for convenience

        self.df['year'] = pd.DatetimeIndex(self.df.index).year #create year column to sort 
        self.df['quarter'] = pd.DatetimeIndex(self.df.index).quarter #create quarter column to sort 
    
        self.df = self.df.sort_index(ascending=True) 
        print(self.df) 
    
    # ---- PART 2: GENERATE MISSING DATA POINTS 
    # because fundamentals are in quarter and stock prices are daily --> must summarise stock prices into quarter summaries
    def generate_quarter_data (self): 
        self.years = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020] #list to iterate (dev) 
        self.quarters = [1,2,3,4] #quarters list to iterate

        #we will store variables generated in the newly created empty list,
        #lists will be passed through into a result table to summarise 
        year_list = [] #year 
        quarter_list = [] #quarter
        r_list = [] #ri score 
        qmean_list = [] #quarter mean 
        x_positions = [] #whether the minima comes before or after the maximum

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
                        q_mean = df_2['Close'].mean() #quarter mean 
                        #append all values to list 
                        r_list.append(r_i)
                        year_list.append(y) 
                        quarter_list.append(q)
                        qmean_list.append(q_mean) 
                    else: #if minima comes after maxima, we want to find min/max-1 (this will signal a negative downturn) 
                        r_i = df_2['Close'].min() / df_2['Close'].max() - 1
                        q_mean = df_2['Close'].mean() #quarter mean 
                        #append all values to list 
                        r_list.append(r_i) 
                        year_list.append(y) 
                        quarter_list.append(q)
                        qmean_list.append(q_mean) 
        
        self.result = pd.DataFrame() #creating empty df to store results 
        #append results from above loop 
        self.result['Year'] = year_list 
        self.result['Quarter'] = quarter_list 
        self.result['Q_Mean'] =  qmean_list 
        self.result['Rq'] = (self.result['Q_Mean'] - self.result['Q_Mean'].shift(1)) / self.result['Q_Mean'].shift(1) #shift quarter mean down 1 to find difference between Qx and Qx-1. This signifies the difference between this quarter and last quarter in terms of returns
        self.result['Ri'] = r_list
        self.result['ArgMax/Min Position'] = x_positions
        self.result = self.result.sort_values(['Year', 'Quarter'], ascending = True) #sort time for better visualisation and fitting into income statement data 
        self.total_period = len(self.result.index) #this total period variable will determine how much data is pulled from fundamentals list
        self.result = self.result.fillna(0) #subjected to rev      
        print(self.result) 

    # ---- PART 4: IMPORT FUNDAMENTALS 
    def import_financials(self): 
        ##RATIOS LIST (see documentation)
        self.ratio_df = tcbs_ticker.ratio(user_ticker, period_type=0, period=self.total_period) #get data 
        self.ratio_df = self.ratio_df.sort_values(['YearReport', 'LengthReport'], ascending=True) #sort values (see result list for sorting order) 
        self.ratios_list = ['revenue', 'operationProfit', 'GrossProfit', 'netProfit', 'InterestExpense', 'SellingExpense', 'GAExpense', 'preTax', 'ebitOnInterest', 'netPayableOnEquity', 'shortOnLongTermPayable', 'payableOnEquity', 'payable', 'asset', 'liability', 'equity', 'STDebt', 'LTDebt', 'capitalize', 'roe'] #create all features list from ratios 

        #here, we are interested only in the YoY return of these features 
        #create empty dataframe as we will only consider the YoY values in our model 
        self.ratios_returns = pd.DataFrame() #create empty dataFrame 
        self.ratios_returns['YearReport'] = self.ratio_df['YearReport'] #set year 
        self.ratios_returns['LengthReport'] = self.ratio_df['LengthReport'] #set quarter 
        #looping through all items in ratios list, we will create a derived YoY return
        #shift(1) because it will show growth between this and last quarter
        for item in self.ratios_list: 
            self.ratios_returns[item+'_YoY'] = (self.ratio_df[item] - self.ratio_df[item].shift(1)) / self.ratio_df[item].shift(1) 
        self.ratios_returns = self.ratios_returns.fillna(0) 
        correlation(self.ratios_returns, 0.8) #run data through the first correlation function to eliminate co-linearity
        print(self.ratios_returns)
        self.ratios = self.ratios_returns.columns.tolist()
        #self.ratios.remove(['priceToEarning_YoY', 'priceToBook_YoY','dividend_YoY']) 
        #this stores new columns for convenient modelling 

        ##BALANCE SHEET (see documentation & workflow on ratios list) 
        self.balancesheet_df = tcbs_fin.finance_balance_sheet(user_ticker, period_type=0, period=self.total_period)
        self.balancesheet_df = self.balancesheet_df.sort_values(['YearReport', 'LengthReport'], ascending=True)
        self.balance_list = ['cash', 'liability', 'equity', 'shortDebt', 'longDebt', 'shortAsset', 'longAsset', 'debt', 'asset', 'shortInvest', 'shortReceivable', 'inventory', 'longReceivable', 'fixedAsset', 'capital']

        self.balance_returns = pd.DataFrame()
        self.balance_returns['YearReport'] = self.balancesheet_df['YearReport']
        self.balance_returns['LengthReport'] = self.balancesheet_df['LengthReport']
        for item in self.balance_list: 
            self.balance_returns[str(item)+'_YoY'] = (self.balancesheet_df[item] - self.balancesheet_df[item].shift(1)) / self.balancesheet_df[item].shift(1) 
        self.balance_returns = self.balance_returns.fillna(0) 
        correlation(self.balance_returns, 0.8) 
        print(self.balance_returns)
        self.balance = self.balance_returns.columns.tolist()

        ##INCOME STATEMENT (see documentation & workflow on ratios list)
        self.income_df = tcbs_fin.finance_income_statement(user_ticker, period_type=0, period=self.total_period) 
        self.income_df = self.income_df.sort_values(['YearReport', 'LengthReport'], ascending=True) 
        self.income_list = ['Revenue', 'Cost_Of_Goods_Sold', 'Gross_Profit', 'Interest_Expenses', 'Operation_Expenses', 'Operating_Profit', 'Pre_Tax_Profit', 'Post_Tax_Profit', 'Net_Income', 'EBITDA']

        self.income_returns = pd.DataFrame()
        self.income_returns['YearReport'] = self.income_df['YearReport']
        self.income_returns['LengthReport'] = self.income_df['LengthReport']
        for item in self.income_list: 
            self.income_returns[str(item)+'_YoY'] = (self.income_df[item] - self.income_df[item].shift(1)) / self.income_df[item].shift(1)
        self.income_returns = self.income_returns.fillna(0) 
        correlation(self.income_returns, 0.8) 
        print(self.income_returns)
        self.income = self.income_returns.columns.tolist()

    # ---- PART 5: BI-VARIATE REGRESSION -- this will run if the 'simple' model option is chosen in command line
    def single_feature(self, feauture_count, mean_option): 
        # if user chose 'ratios' 
        if features_pool == 'ratios': 
            features_list = self.ratios[2:] #this ignores the first two columns, which are year and quarter reported 
            #create empty lists to store final results 
            feature_list = [] 
            score_list = [] 
            ticker_list = []

            #looping through all ratios in list 
            for f in features_list: 
                features = self.ratios_returns[[f]] #features 
                
                outcome = self.result[[mean_option]] #outcomes 

                x_train, x_test, y_train, y_test = train_test_split(features, outcome, train_size = 0.8, shuffle=False) #splitting dataset for better testing 

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

                #appending results to new dataFrame for better analysis 
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
        
        #if user chose 'balance', flow similar to ratios 
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
        
        #if user chose 'income', flow similar to ratios 
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

    #PART 5 ---- AGGREGATE REGRESSION MODELLING - runs if user chooses 'aggregrate' 
    def multi_features(self, features_pool, mean_option): 
        #the powerset function will parse through a list and return all the possible combinations of variables within the list 
        #because we are interested in finding both maximising the count of features, as well as the combination of specific features, explore dataset with this 
        #conclusion is made upon CSV exported from this 
        def powerset(iterable): 
            select = list(iterable)  # allows duplicate elements
            return chain.from_iterable(combinations(select, r) for r in range(len(select)+1))

        #if user select 'ratios' pool: 
        if features_pool == 'ratios': 
            ratio_pool = self.ratios[2:]

            combo_list = [] #list to receive combinations 

            for combo in enumerate(powerset(ratio_pool), 1): #return in combos from the powerset function 
                f = combo
                combo_list.append(f[1]) #aggregate into list 

            feature_list = [] 
            count_list = [] 
            score_list = [] 
            ticker_list = []
            coefs_list = []
            intercepts_list = []

            for c in combo_list[1:]: #iterating through the entire combinations list. as combo_list begins with an empty list, start from index 1 
                x = list(c) #because items are stored under tuples, turn into list 
                features = self.ratios_returns[x] #declaring features 
                outcome = self.result[[mean_option]] #declaring outomces 
        
                x_train, x_test, y_train, y_test = train_test_split(features, outcome, train_size = 0.8, shuffle=False) #splitting train and test sets 

                model = LinearRegression() #create model 
                model.fit(x_train, y_train) #fit model 
                score = model.score(x_train, y_train) #score train model 
                score_test = model.score(x_test, y_test) #score test model 
                predictions = model.predict(x_train) #pass x-train through predict to determine best fit plane 
                coefs = model.coef_ #calculate coefficient - multiplicative factor of each feature to outcome
                intercepts = model.intercept_ #calculate intercept
                print(coefs, intercepts) 

                #append results to empty dataFrame 
                total = pd.DataFrame() 
                ticker_list.append(user_ticker) 
                total['ticker'] = ticker_list 
                print(user_ticker) 
                feature_list.append(c) 
                total['feature'] = feature_list
                print(c)
                count_list.append(len(list(x))) 
                total['features_count'] = count_list 
                score_list.append(score) 
                total['score'] = score_list
                print(score)
                coefs_list.append(coefs) 
                total['coef'] = coefs_list 
                intercepts_list.append(intercepts) 
                total['intercept'] = intercepts_list
                total = total.sort_values('score', ascending=False)
                total.to_csv(user_path+user_ticker+'/ratio_agg.csv')
        
        #if user select 'balance' pool: 
        if features_pool == 'balance': 
            balance_pool = self.balance[2:]

            combo_list = [] 

            for combo in enumerate(powerset(balance_pool), 1):
                f = combo
                combo_list.append(f[1])

            feature_list = [] 
            count_list = []
            score_list = [] 
            ticker_list = []
            coefs_list = [] 
            intercepts_list =[]

            for c in combo_list[1:]: 
                x = list(c) 
                features = self.balance_returns[x]
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
                count_list.append(len(list(x))) 
                total['features_count'] = count_list 
                score_list.append(score) 
                total['score'] = score_list
                print(score)
                coefs_list.append(coefs) 
                total['coef'] = coefs_list 
                intercepts_list.append(intercepts) 
                total['intercept'] = intercepts_list
                total = total.sort_values('score', ascending=False)
                total.to_csv(user_path+user_ticker+'/balance_agg.csv')

        #if user select 'income' pool: 
        if features_pool == 'income': 
            income_pool = self.income[2:]

            combo_list = [] 

            for combo in enumerate(powerset(income_pool), 1):
                f = combo
                combo_list.append(f[1])

            feature_list = [] 
            count_list = []
            score_list = [] 
            ticker_list = []
            coefs_list = [] 
            intercepts_list =[]

            for c in combo_list[1:]: 
                x = list(c)
                features = self.income_returns[x]
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
                count_list.append(len(list(x))) 
                total['features_count'] = count_list 
                score_list.append(score) 
                total['score'] = score_list
                print(score)
                coefs_list.append(coefs) 
                total['coef'] = coefs_list 
                intercepts_list.append(intercepts) 
                total['intercept'] = intercepts_list
                total = total.sort_values('score', ascending=False)
                total.to_csv(user_path+user_ticker+'/income_agg.csv')

new = optimising_features_score() 
new.clean_data() 
new.generate_quarter_data() 
new.import_financials() 
if user_mode == 'simple': 
    new.single_feature(features_pool, user_mean)
elif user_mode == 'aggregate': 
    new.multi_features(features_pool, user_mean) 
else: 
    print("Error")