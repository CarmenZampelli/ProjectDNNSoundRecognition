import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
import re
import os


path='C:/Users/carme/Desktop/FSD50K dataset/unsplit/SaveEmbeddings/'


train_df=pd.read_csv('dataframe_train')
print(train_df)

Min=10
Max=30
train_df_selected= train_df[(train_df["nframe"]>Min) & (train_df["nframe"]<Max)]


X_train=[]
for i in train_df_selected['0']:
    s=np.load('C:/Users/carme/Desktop/FSD50K dataset/unsplit/SaveEmbeddings/train/'+i)
    X_train.append(s)
    
y_train=train_df_selected['labels']

train_df_selected.head()


#validation
val_df=pd.read_csv('dataframe_val')

val_df_selected=val_df[(val_df['nframe']>Min) & (val_df['nframe']<Max)]

X_val=[]
for i in val_df_selected['0']:
    s=np.load('C:/Users/carme/Desktop/FSD50K dataset/unsplit/SaveEmbeddings/validation/'+i)
    X_val.append(s)

y_val=val_df_selected['labels']
val_df_selected.head()


#test
test_df=pd.read_csv('dataframe_test')

test_df_selected=test_df[(test_df['nframe']>Min) & (test_df['nframe']<Max)]
X_test=[]
for i in test_df_selected['0']:
    s=np.load('C:/Users/carme/Desktop/FSD50K dataset/unsplit/SaveEmbeddings/test/'+i)
    X_test.append(s)

y_test=test_df_selected['labels']
test_df_selected.head()


#one hot encoding
y_tot=np.array(pd.concat([y_train, y_val, y_test], axis =0))

from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

lb = LabelEncoder()
y_tot_lb = to_categorical(lb.fit_transform(y_tot))


y_train_lb=y_tot_lb[0:len(y_train)]
y_val_lb=y_tot_lb[len(y_train):len(y_val)+len(y_train)]
y_test_lb=y_tot_lb[len(y_val)+len(y_train):len(y_val)+len(y_train)+len(y_test)]

from keras.preprocessing.sequence import pad_sequences

maxlen=200
y_train_ml=pad_sequences(y_train_lb, maxlen=maxlen)
y_val_ml=pad_sequences(y_val_lb, maxlen=maxlen)
y_test_ml=pad_sequences(y_test_lb, maxlen=maxlen)


#from sklearn.preprocessing import StandardScaler
#ss = StandardScaler()
#X_train_1=[]
#for i in X_train:
 #   s=ss.fit_transform(i)
  #  X_train_1.append(s)
#X_val_1=[]
#for i in X_val:
 #   s=ss.fit_transform(i)
  #  X_val_1.append(s)
#X_test_1=[]
#for i in X_test:
 #   s=ss.fit_transform(i)
  #  X_test_1.append(s)


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Activation, Dense, RepeatVector
from tensorflow.keras import Sequential


#build model
inputs = keras.Input(shape=(None,1024))
mask=layers.Masking(mask_value=0, dtype=float)(inputs)
LSTM=layers.LSTM(32, return_sequences=False)(mask)
output=layers.Dense(200, activation='softmax')(LSTM)
model = keras.Model(inputs=inputs, outputs=output, name="LSTM")
model.summary()


 

y_train_input=tf.keras.preprocessing.sequence.pad_sequences(y_train_ml, padding="post")
y_val_input=tf.keras.preprocessing.sequence.pad_sequences(y_val_ml, padding="post")
y_test_input=tf.keras.preprocessing.sequence.pad_sequences(y_test_ml, padding="post")


time_step=[]
for n in range(0,len(X_train)):
    p=X_train[n].shape
    time_step.append(p[0])


max=0
for i in time_step:
    if i>max:
        max=i
print(max)

#padding 
X_train_d1=[]
for i in X_train:
    len=max-(i.shape)[0]
    l=np.pad(i, [(0,len), (0,0)],mode='constant', constant_values=0)
    X_train_d1.append(l)


X_in_train=tf.convert_to_tensor(X_train_d1)


#padding 
X_val_d1=[]
for i in X_val:
    len=max-(i.shape)[0]
    l=np.pad(i, [(0,len), (0,0)],mode='constant', constant_values=0)
    X_val_d1.append(l)

X_in_val=tf.convert_to_tensor(X_val_d1)


#padding 
X_test_d1=[]
for i in X_test:
    len=max-(i.shape)[0]
    l=np.pad(i, [(0,len), (0,0)],mode='constant', constant_values=0)
    X_test_d1.append(l)

X_in_test=tf.convert_to_tensor(X_test_d1)


opt=tf.keras.optimizers.Adam(learning_rate=0.00001)    
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
history = model.fit(X_in_train, y_train_input, batch_size=10, epochs=15, validation_data=(X_in_val, y_val_input))


test_scores = model.evaluate(X_in_test, y_test_input, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

