#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.utils import resample
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
from keras.models import load_model

#чтение файлов
df = pd.read_csv('data.csv')
a = df.describe()
df.info()
# добавляем целевую переменную(nextCycleBreak)
x = df['cycle'].values
b = [0] * len(df['cycle']) 
for i in range(len(x)-1):
    if  x[i] > x[i+1]:
        b[i] = 1
b[-1] = 1
df['nextCycleBreak'] = b
# добавляем новую переменную(количество циклов до поломки RUL)
rul = pd.DataFrame(df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
df = df.merge(rul, on=['id'], how='left')
df['RUL'] = df['max'] - df['cycle']
df.drop('max', axis=1, inplace=True)
df.head()
#исключаем константные переменные 
print(a)
df = df.drop(['p00','p01','p07','p09','p10','p16','p17'],axis=1)
X = df[:11770]
Xt = df[11770:]
#-----------------------Подготовка данных----------
#--------------------------------------------------
# Приминяем scale к нашим данным кроме некоторых переменных
cols_normalize = X.columns.difference(['id','cycle','nextCycleBreak','RUL'])
min_max_scaler = preprocessing.MinMaxScaler()
norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(X[cols_normalize]), 
                             columns=cols_normalize, 
                             index=X.index)
join_df = X[X.columns.difference(cols_normalize)].join(norm_train_df)
X = join_df.reindex(columns = X.columns)
X.head()
X = X.sort_values(by = ['id','cycle'])
#scale для тестовой выборки
y_train = X.nextCycleBreak
X_train = X.drop('nextCycleBreak', axis = 1)
norm_test_df = pd.DataFrame(min_max_scaler.transform(Xt[cols_normalize]), 
                            columns=cols_normalize, 
                            index=Xt.index)
test_join_df = Xt[Xt.columns.difference(cols_normalize)].join(norm_test_df)
Xt = test_join_df.reindex(columns = Xt.columns)
Xt = Xt.reset_index(drop=True)
Xt = Xt.sort_values(by = ['id','cycle'])
Xt.head()
y_test = Xt.nextCycleBreak
X_test = Xt.drop('nextCycleBreak', axis = 1)

#Подготавливаем данные для time-series модели
#функция генерации последовательности
def gen_sequence(id_df, seq_length, seq_cols):
    df_zeros=pd.DataFrame(np.zeros((seq_length-1,id_df.shape[1])),columns=id_df.columns)
    id_df=df_zeros.append(id_df,ignore_index=True)
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array=[]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        lstm_array.append(data_array[start:stop, :])
    return np.array(lstm_array)

# функция генерации label
def gen_label(id_df, seq_length, seq_cols,label):
    df_zeros=pd.DataFrame(np.zeros((seq_length-1,id_df.shape[1])),columns=id_df.columns)
    id_df=df_zeros.append(id_df,ignore_index=True)
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    y_label=[]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        y_label.append(id_df[label][stop])
    return np.array(y_label)
# параметры обучающих наборов
seq_length = 50
h =list(df.columns)
seq_columns = h[2:18]
label_col = 'nextCycleBreak'

# генерация обучающих наборов
X_train=np.concatenate(list(list(gen_sequence(X[X['id']==id], seq_length, seq_columns)) for id in X['id'].unique()))
print(X_train.shape)

y_train=np.concatenate(list(list(gen_label(X[X['id']==id], 50, seq_columns,'nextCycleBreak')) for id in X['id'].unique()))
print(y_train.shape)

X_test=np.concatenate(list(list(gen_sequence(Xt[Xt['id']==id], seq_length, seq_columns)) for id in Xt['id'].unique()))
print(X_test.shape)

y_test=np.concatenate(list(list(gen_label(Xt[Xt['id']==id], 50, seq_columns,'nextCycleBreak')) for id in Xt['id'].unique()))
print(y_test.shape)
#-----------------------------Инициализация модели----------------------
#------------------------------------------------------------------------
"""
nb_features =X_train.shape[2]
timestamp=seq_length

model = Sequential()

model.add(LSTM(
         input_shape=(timestamp, nb_features),
         units=50,
         return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(
          units=25,
          return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
# fit the network
"""
model = load_model('my_model.h5')
"""
model.fit(X_train, y_train, epochs=100, batch_size=200, validation_split=0.05, verbose=1,class_weight = {0:1,1:18},
          callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')])
"""
y_pred=model.predict_classes(X_test)
print('Accuracy of model on test data: ',accuracy_score(y_test,y_pred))
print('Confusion Matrix: \n',confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred))
#model.save('my_model.h5')
#---------------------------------------------------
