import quandl
import pandas as pd

quandl.ApiConfig.api_key = 'KF72u6k4HWWrTg-4PjTP'

def get_data(ticker, date_from, date_to):
    data = quandl.get_table('IFT/NSA', qopts = { 'columns': ['ticker', 'date', 'sentiment', 'sentiment_high', 'sentiment_low', 'news_volume', 'news_buzz'] }, ticker = [ticker], date = { 'gte': date_from, 'lte': date_to }, exchange_cd = 'US')
    price_data = quandl.get_table('WIKI/PRICES', qopts = { 'columns': ['ticker', 'date', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume'] }, ticker = [ticker], date = { 'gte': date_from, 'lte': date_to })
    
    # Social media platform
    twitter_data = quandl.get_table('SMA/TWTD', qopts = { 'columns': ['brand_ticker', 'date', 'followers_count','engagement_score'] }, brand_ticker = ticker, date = { 'gte': date_from, 'lte': date_to }, paginate=True)
    
    insta_data = quandl.get_table('SMA/INSD', qopts = { 'columns': ['brand_ticker', 'geography', 'date', 'followers_count', 'engagement_score'] }, brand_ticker = ticker, date = { 'gte': date_from, 'lte': date_to }, paginate=True)
    
    core_data = quandl.get_table('SHARADAR/SF1', dimension='ARQ', ticker=ticker, qopts = { 'columns': ['ticker', 'calendardate', 'fcf', 'epsdil', 'ebitdamargin']}, calendardate = { 'gte': date_from, 'lte': date_to })
    fb_data = quandl.get_table('SMA/FBD', brand_ticker = [ticker], qopts = { 'columns': ['brand_ticker', 'date', 'fans', 'engagement_score', 'people_talking_about'] }, date = { 'gte': date_from, 'lte': date_to },paginate=True)

    def clean():    
        def organise_data(dataframe):
            a = dataframe.groupby(['date'])['followers_count', 'engagement_score'].sum()
            a.reset_index(inplace=True)
            a['ticker'] = dataframe['brand_ticker']
            #a['sector'] = dataframe['sector']
            a = a[['date', 'ticker', 'followers_count', 'engagement_score']]
            
            return a
        
        a_d = organise_data(twitter_data)
        b_d = organise_data(insta_data)
    
        def merge_sent_twitter(dataframe, dataframe2):
            c = dataframe.merge(dataframe2, left_on=['date', 'ticker'], right_on=['date', 'ticker'], how='outer')
            return c
        
        def merge_combo_insta(dataframe, dataframe2):
            d = dataframe.merge(dataframe2, left_on=['date', 'ticker'], right_on=['date', 'ticker'], how='outer')
            return d
    
        new = merge_sent_twitter(data, a_d)
        
        new_new = merge_combo_insta(new, b_d)
    
        def merge_combo2_price(dataframe, dataframe2):
            e = dataframe.merge(dataframe2, left_on=['date', 'ticker'], right_on=['date', 'ticker'], how='outer')
            return e
    
        final = merge_combo2_price(new_new, price_data)
    
        def organise_fb_data(fb_data):
          fb_data.rename(columns= {'brand_ticker':'ticker'},inplace=True)
          fb_data2 = fb_data.groupby(['date'])['fans', 'engagement_score', 'people_talking_about'].sum()
          fb_data2.reset_index(inplace=True)
          fb_data2['ticker'] = fb_data['ticker']
          #fb_data2['sector'] = fb_data['sector']
          fb_data2 = fb_data2[['date', 'ticker', 'fans', 'engagement_score', 'people_talking_about']]
          return fb_data2
        
        def merge_data(fb_data2, core_data):
           core_data.rename(columns = {'calendardate':'date'},inplace=True)
           total_data = fb_data2.merge(core_data, left_on=['date', 'ticker'], right_on=['date', 'ticker'], how='outer')
           return total_data
        
        fb_data2 = organise_fb_data(fb_data)
        mini_overall = merge_data(fb_data2, core_data)
        
        def final_merge(dataframe, dataframe2):
            the_final = dataframe.merge(dataframe2, left_on=['date', 'ticker'], right_on=['date', 'ticker'], how='outer')
            return the_final
        
        the_final = final_merge(final, mini_overall)
        return the_final
    
    final_data = clean()
    final_data['adj_close'].fillna(method='ffill', inplace=True)
    final_data = final_data
    final_data['up_or_down'] = final_data['adj_close'] - final_data['adj_close'].shift(1)
    final_data['up_or_down'] = final_data['up_or_down'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    final_data['fcf'].fillna(method='ffill', inplace=True)
    final_data['epsdil'].fillna(method='ffill', inplace=True)
    final_data['ebitdamargin'].fillna(method='ffill', inplace=True)
    final_data = final_data[['ticker', 'date', 'sentiment', 'sentiment_high', 'sentiment_low', 'news_volume', 'news_buzz', 'followers_count_x', 'engagement_score_x', 'followers_count_y', 'engagement_score_y', 'adj_close', 'fans', 'engagement_score', 'people_talking_about', 'fcf', 'epsdil', 'ebitdamargin', 'up_or_down']]
    
    return final_data

#the_final3 = get_data('AAPL', '2013-12-31', '2017-12-31')
#the_final3.to_csv('aggregate3.csv', index=False)

data = []
for i in range(len(list_equities)):
    print(i)
    equity = list_equities[i]
    temp = get_data(equity, '2013-12-31', '2018-12-31')
    temp.sort_values(by='date', ascending=True, inplace=True)
    data.append(temp)
    #temp.to_csv(equity + ".csv", index=False)
    
overall = pd.concat(data)
overall = overall[1:]

overall.to_csv('data2.csv', index=False)
    
overall.dropna(inplace=True)

import pandas as pd

df = pd.read_csv('data2.csv', engine='python')
df['up_or_down'].value_counts()

df = df[df['up_or_down'] != 0]

# In-house libraries
import machine_learning as ml

# Scikit learn evaluation metrics and splitting dataset
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Machine learning libraries
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

# Splitting dataset into train and test set
X = df[["sentiment", 
"sentiment_high",
"sentiment_low",
"sentiment_volatility",
"news_volume",
"news_buzz", 
"followers_count_x", 
"engagement_score_x", 
"followers_count_y", 
"engagement_score_y",
"fans", 
"engagement_score", 
"people_talking_about", 
"fcf", 
"epsdil", 
"ebitdamargin"]]
Y = df["up_or_down"]

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3, random_state=42)

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

# ML estimators and parameters
estimators = [BernoulliNB(),
              LinearSVC(),
              LogisticRegression(),
              RandomForestClassifier(n_estimators=40, n_jobs=-1),
              ExtraTreesClassifier(n_estimators=40, n_jobs=-1),
              GradientBoostingClassifier(n_estimators=40, max_depth=7, learning_rate=1, random_state = 0),
              DecisionTreeClassifier()]
              #MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)]

# Train and save machine learning models
ml.train_ML_models(estimators, X_train, y_train, X_test, y_test)

def ML_load(model):
    """
    Load trained machine learning model
    """
    loaded_model = pickle.load(open(model,'rb'))
    
    return loaded_model

BernoulliNB = ml.ML_load('BernoulliNB10.sav')
LinearSVC = ml.ML_load('LinearSVC10.sav')
LogisticRegression = ml.ML_load('LogisticRegression10.sav')
RandomForestClassifier = ml.ML_load('RandomForestClassifier10.sav')
ExtraTreesClassifier = ml.ML_load('ExtraTreesClassifier10.sav')
GradientBoostingClassifier = ml.ML_load('GradientBoostingClassifier10.sav')
DecisionTreeClassifier = ml.ML_load('DecisionTreeClassifier10.sav')

df2 = pd.read_csv('2018_test_dataset.csv', engine='python')
df2 = df2[df2['up_or_down'] != 0]
df2.to_csv('2018_test_dataset3.csv', index=False)

df2['BernoulliNB'] = BernoulliNB.predict(df2[["sentiment", "sentiment_high", "sentiment_low", "sentiment_volatility", "news_volume", "news_buzz",  "followers_count_x", "engagement_score_x", "followers_count_y", "engagement_score_y", "fans", "engagement_score", "people_talking_about", "fcf", "epsdil", "ebitdamargin"]])
df2['LinearSVC'] = LinearSVC.predict(df2[["sentiment", "sentiment_high", "sentiment_low", "sentiment_volatility", "news_volume", "news_buzz",  "followers_count_x", "engagement_score_x", "followers_count_y", "engagement_score_y", "fans", "engagement_score", "people_talking_about", "fcf", "epsdil", "ebitdamargin"]])
df2['LogisticRegression'] = LogisticRegression.predict(df2[["sentiment", "sentiment_high", "sentiment_low", "sentiment_volatility", "news_volume", "news_buzz",  "followers_count_x", "engagement_score_x", "followers_count_y", "engagement_score_y", "fans", "engagement_score", "people_talking_about", "fcf", "epsdil", "ebitdamargin"]])
df2['RandomForestClassifier'] = RandomForestClassifier.predict(df2[["sentiment", "sentiment_high", "sentiment_low", "sentiment_volatility", "news_volume", "news_buzz",  "followers_count_x", "engagement_score_x", "followers_count_y", "engagement_score_y", "fans", "engagement_score", "people_talking_about", "fcf", "epsdil", "ebitdamargin"]])
df2['ExtraTreesClassifier'] = ExtraTreesClassifier.predict(df2[["sentiment", "sentiment_high", "sentiment_low", "sentiment_volatility", "news_volume", "news_buzz",  "followers_count_x", "engagement_score_x", "followers_count_y", "engagement_score_y", "fans", "engagement_score", "people_talking_about", "fcf", "epsdil", "ebitdamargin"]])
df2['GradientBoostingClassifier'] = GradientBoostingClassifier.predict(df2[["sentiment", "sentiment_high", "sentiment_low", "sentiment_volatility", "news_volume", "news_buzz",  "followers_count_x", "engagement_score_x", "followers_count_y", "engagement_score_y", "fans", "engagement_score", "people_talking_about", "fcf", "epsdil", "ebitdamargin"]])
df2['DecisionTreeClassifier'] = DecisionTreeClassifier.predict(df2[["sentiment", "sentiment_high", "sentiment_low", "sentiment_volatility", "news_volume", "news_buzz",  "followers_count_x", "engagement_score_x", "followers_count_y", "engagement_score_y", "fans", "engagement_score", "people_talking_about", "fcf", "epsdil", "ebitdamargin"]])