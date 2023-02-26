
import re, string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from gensim.models import Word2Vec




import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (RandomForestClassifier,
                              AdaBoostClassifier,
                              GradientBoostingClassifier,
                              HistGradientBoostingClassifier)

import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


data_train= pd.read_csv('/Users/bhavikachilamkurthy/Downloads/crime_train.csv')
data_test=pd.read_csv('/Users/bhavikachilamkurthy/Downloads/crime_test.csv')
data_train['key'] = 'train'
data_test['CRIMETYPE'] = 'null'
data_test['key'] = 'test'

data_train = data_train.append(data_test, ignore_index=True)


A=data_train['CRIMETYPE'].value_counts()




data_train = data_train.dropna(subset=['NARRATIVE'])



# Calculating the word count
data_train['word_count'] = data_train['NARRATIVE'].apply(lambda x: len(str(x).split()))


# Moving the data to lowercase and removing punctuations
def initialization(sentences_data):

    sentences_data = sentences_data.lower()
    sentences_data = sentences_data.strip()
    sentences_data = re.compile('<.*?>').sub('', sentences_data)
    sentences_data = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', sentences_data)
    sentences_data = re.sub('\s+', ' ', sentences_data)
    sentences_data = re.sub(r'\[[0-9]*\]', ' ', sentences_data)
    sentences_data = re.sub(r'[^\w\s]', '', str(sentences_data).lower().strip())
    sentences_data = re.sub(r'\d', ' ', sentences_data)
    sentences_data = re.sub(r'\s+', ' ', sentences_data)
    return sentences_data



def Endings(string):
    a = [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)



M = WordNetLemmatizer()



def pos_net(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN



def middle_process(string):
    mag = nltk.pos_tag(word_tokenize(string))
    a = [M.lemmatize(tag[0], pos_net(tag[1])) for idx, tag in
         enumerate(mag)]
    return " ".join(a)

def pro_last(string):
    return middle_process(Endings(initialization(string)))

data_train['clean_text'] = data_train['NARRATIVE'].apply(lambda x: pro_last(x))


data_test = data_train[data_train['key'] == 'test']
data_train = data_train[data_train['key'] == 'train']


A_train = data_train['NARRATIVE'].squeeze()
A_test = data_test['NARRATIVE'].squeeze()
B_train = data_train['CRIMETYPE'].squeeze()
B_test = data_test['CRIMETYPE'].squeeze()


A_traintoken= [nltk.word_tokenize(i) for i in A_train]
A_testtoken= [nltk.word_tokenize(i) for i in A_test]

# Generating Tf-Idf vectors
tfidf_vectorizer = TfidfVectorizer(use_idf=True)
X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(A_train)
X_test_vectors_tfidf = tfidf_vectorizer.transform(A_test)




#CatBoost Model
cb = CatBoostClassifier(n_estimators=2000,
                        colsample_bylevel=0.06,
                        max_leaves=31,
                        subsample=0.67,
                        verbose=0,
                        thread_count=6,
                        random_state=1234)
cb.fit(X_train_vectors_tfidf, B_train)

y_predict = cb .predict(X_test_vectors_tfidf)
# predict_v = np.where(y_predict == 1, 'BURG', 'BTFV')
data_test['id'] = pd.to_numeric(data_test['id'])
result = pd.DataFrame({'id': data_test.id, 'CRIMETYPE': y_predict})
result.to_csv("/Users/bhavikachilamkurthy/Downloads/CatBoostClassifier.csv", index=False)
#pd.DataFrame(y_predict).to_csv("/Users/bhavikachilamkurthy/Downloads/CatBoostClassifier.csv")
print("finished CatBoost Model execution")

#LGBM Model

lgbm = LGBMClassifier(n_estimators=2000,
                      feature_fraction=0.06,
                      bagging_fraction=0.67,
                      bagging_freq=1,
                      verbose=0,
                      n_jobs=6,
                      random_state=1234)
lgbm.fit(X_train_vectors_tfidf, B_train)
y_predict = lgbm .predict(X_test_vectors_tfidf)
#predict_v = np.where(y_predict == 1, 'BURG', 'BTFV')
data_test['id'] = pd.to_numeric(data_test['id'])
result = pd.DataFrame({'id': data_test.id, 'CRIMETYPE': y_predict})
result.to_csv("/Users/bhavikachilamkurthy/Downloads/LGBMClassifier.csv", index=False)
#pd.DataFrame(y_predict).to_csv("/Users/bhavikachilamkurthy/Downloads/LGBMClassifier.csv")
print("finished LGBM Model execution")

#RandomForest Model

rf = RandomForestClassifier(n_estimators=500,
                            max_features=0.06,
                            n_jobs=6,
                            random_state=1234)
rf.fit(X_train_vectors_tfidf, B_train)
y_predict = rf.predict(X_test_vectors_tfidf)
#predict_v = np.where(y_predict == 1, 'BURG', 'BTFV')
data_test['id'] = pd.to_numeric(data_test['id'])
result = pd.DataFrame({'id': data_test.id, 'CRIMETYPE': y_predict})
result.to_csv("/Users/bhavikachilamkurthy/Downloads/rfclassifier.csv", index=False)
#pd.DataFrame(y_predict).to_csv("/Users/bhavikachilamkurthy/Downloads/rfclassifier.csv")

print("finished RF Model execution")
base_estim = DecisionTreeClassifier(max_depth=1, max_features=0.06)
#AdaBoost Model
ab = AdaBoostClassifier(base_estimator=base_estim,
                        n_estimators=500,
                        learning_rate=0.5,
                        random_state=1234)

ab.fit(X_train_vectors_tfidf, B_train)
y_predict = ab.predict(X_test_vectors_tfidf)
#predict_v = np.where(y_predict == 1, 'BURG', 'BTFV')
data_test['id'] = pd.to_numeric(data_test['id'])
result = pd.DataFrame({'id': data_test.id, 'CRIMETYPE': y_predict})
result.to_csv("/Users/bhavikachilamkurthy/Downloads/ab.csv", index=False)

#pd.DataFrame(y_predict).to_csv("/Users/bhavikachilamkurthy/Downloads/ab.csv")
print("finished Ada Boost Model execution")

#GradientBoost Model

gbm = GradientBoostingClassifier(n_estimators=2000,
                                 subsample=0.67,
                                 max_features=0.06,
                                 validation_fraction=0.1,
                                 n_iter_no_change=15,
                                 verbose=0,
                                 random_state=1234)

gbm.fit(X_train_vectors_tfidf, B_train)
y_predict = gbm.predict(X_test_vectors_tfidf)
#predict_v = np.where(y_predict == 1, 'BURG', 'BTFV')
data_test['id'] = pd.to_numeric(data_test['id'])
result = pd.DataFrame({'id': data_test.id, 'CRIMETYPE': y_predict})
result.to_csv("/Users/bhavikachilamkurthy/Downloads/gbm.csv", index=False)
#pd.DataFrame(y_predict).to_csv("/Users/bhavikachilamkurthy/Downloads/gbm.csv")
print("finished GradientBoost Model execution")

#XGBoost Model
xgb = XGBClassifier(n_estimators=2000,
                    tree_method='hist',
                    subsample=0.67,
                    colsample_level=0.06,
                    verbose=0,
                    n_jobs=6,
                    random_state=1234)

xgb.fit(X_train_vectors_tfidf, B_train)
y_predict = xgb.predict(X_test_vectors_tfidf)
#predict_v = np.where(y_predict == 1, 'BURG', 'BTFV')
data_test['id'] = pd.to_numeric(data_test['id'])
result = pd.DataFrame({'id': data_test.id, 'CRIMETYPE': y_predict})
result.to_csv("/Users/bhavikachilamkurthy/Downloads/xgb.csv", index=False)
#pd.DataFrame(y_predict).to_csv("/Users/bhavikachilamkurthy/Downloads/xgb.csv")
print("finished XG Boost Model execution")





