import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import params as yamnet_params
import yamnet as yamnet_model
import tensorflow as tf
from scipy.io import wavfile
import scipy
from scipy import signal
import os
import pandas as pd
import keras


path='C:/Users/carme/Desktop/FSD50K dataset/'
dataframe=pd.read_csv(path+'unsplit/rete_ef/dataframe_train_ef.csv')

base_data_path=path+'unsplit/FSD50K.dev_audio/'

# read data frames
train_df=pd.read_csv(path+'unsplit/dataframe_train.csv')
val_df=pd.read_csv(path+'unsplit/dataframe_val.csv')
test_df=pd.read_csv(path+'unsplit/dataframe_test.csv')


#load embeddings

X_train=[]
filelist_train = train_df['filename']


for filename in filelist_train:
    s=np.load('C:/Users/carme/Desktop/FSD50K dataset/unsplit/SaveEmbeddings/train/'+str(filename)+'.npy')
    X_train.append(s)

y_train=np.array(train_df['label'])


# val data
filelist_val = val_df['filename']

X_val=[]

for filename in filelist_val:
    s=np.load('C:/Users/carme/Desktop/FSD50K dataset/unsplit/SaveEmbeddings/validation/'+str(filename)+'.npy')
    X_val.append(s)

y_val=np.array(val_df['label'])



#test
filelist_test = test_df['filename']
X_test=[]

for filename in filelist_test:
    s=np.load('C:/Users/carme/Desktop/FSD50K dataset/unsplit/SaveEmbeddings/test/'+str(filename)+'.npy')
    X_test.append(s)

y_test=np.array(test_df['label'])



###########################################################################
#pre processing 
#one-hot encoding
vocabulary=pd.read_csv('vocabularyFSD50K.csv')

vocabulary=vocabulary['label']
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


vocabulary
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(vocabulary)
print(integer_encoded)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
######################################################################################
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

# build one-hot encoding matrix for all labels together and then separate in training, val, test
lb = LabelEncoder()
num_labels=lb.fit_transform(vocabulary)
[labels,index] = np.unique(vocabulary, return_index=True)
vocabulary_cat = to_categorical(num_labels)
vocabulary_cat[0]
###################################################################################
voc=pd.DataFrame(onehot_encoded)


voc['labels']=vocabulary
voc.to_csv('vocabulary_one_hot.csv')

voc=pd.read_csv('vocabulary_one_hot.csv')
voc
voc=voc.drop(voc.columns[[0]], axis=1)

Y_train=[]
aa=[]
words=''

for n in range(0, len(y_train)):
    aa=[]
    words=''
    for i in range(0, len(y_train[n])):
      if y_train[n][i]!=",":
          words=words+y_train[n][i]
     
      else: 
         aa.append(words)
         words=''
    aa.append(words)
    Y_train.append(aa)

one_hot=np.zeros(200)
Y_train_oh=[]
#Y_train[2]
for i in Y_train:
    one_hot=np.zeros(200)
    for n in range(0, len(i)):
       for j in range(0, len(voc)):
           if i[n]!=voc['labels'][j]:
               j=j+1
            
           else: 
               one_hot=one_hot+voc.iloc[j, 0:200]
    Y_train_oh.append(one_hot)


y_train_ohe=np.array(Y_train_oh) 


#validation
Y_val=[]
aa_v=[]
words_v=''

for n in range(0, len(y_val)):
    aa_v=[]
    words_v=''
    for i in range(0, len(y_val[n])):
      if y_val[n][i]!=",":
          words_v=words_v+y_val[n][i]
      
      else: 
         aa_v.append(words_v)
         words_v=''
    aa_v.append(words_v)
    Y_val.append(aa_v)


one_hot_v=np.zeros(200)
Y_val_oh=[]

for i in Y_val:
    one_hot_v=np.zeros(200)
    for n in range(0, len(i)):
       for j in range(0, len(voc)):
           if i[n]!=voc['labels'][j]:
               j=j+1
            
           else: 
               one_hot_v=one_hot_v+voc.iloc[j, 0:200]
    Y_val_oh.append(one_hot_v)


y_val_ohe=np.array(Y_val_oh) 

#test
Y_test=[]
aa_t=[]
words_t=''

for n in range(0, len(y_test)):
    aa_t=[]
    words_t=''
    for i in range(0, len(y_test[n])):
      if y_test[n][i]!=",":
          words_t=words_t+y_test[n][i]
      
      else: 
         aa_t.append(words_t)
         words_t=''
    aa_t.append(words_t)
    Y_test.append(aa_t)
 


one_hot_t=np.zeros(200)
Y_test_oh=[]

for i in Y_test:
    one_hot_t=np.zeros(200)
    for n in range(0, len(i)):
       for j in range(0, len(voc)):
           if i[n]!=voc['labels'][j]:
               j=j+1
            
           else: 
               one_hot_t=one_hot_t+voc.iloc[j, 0:200]
    Y_test_oh.append(one_hot_t)
    

y_test_ohe=np.array(Y_test_oh) 
    
#create a list (xx_t) and insert each frame of the X_train 
from numpy import ones

xx_t=[]
len_x=[]

for i in X_train:
    l=len(i)
    for n in i:
        xx_t.append(n)
    l=len(i)
    len_x.append(l)

#replicate y     
yy_t=[]
n=0
for i in y_train_ohe:
    x=ones((len_x[n],1))*i
    yy_t.append(x)
    n=n+1
#insert each element of y in yy list
yy=[]
for i in yy_t:
    for n in i:
        yy.append(n)


x_c=tf.convert_to_tensor(xx_t)
y_c=tf.convert_to_tensor(yy)




#validation
xx_v=[]
len_v=[]
for i in X_val:
    l=len(i)
    for n in i:
        xx_v.append(n)
    l=len(i)
    len_v.append(l)

yy_v=[]
n=0
for i in y_val_ohe:
    x=ones((len_v[n],1))*i
    yy_v.append(x)
    n=n+1
yy_vl=[]
for i in yy_v:
    for n in i:
        yy_vl.append(n)
 
        
x_c_v=tf.convert_to_tensor(xx_v)
y_c_v=tf.convert_to_tensor(yy_vl)

#test      
xx_test=[]
len_t=[]
for i in X_test:
    l=len(i)
    for n in i:
        xx_test.append(n)
    l=len(i)
    len_t.append(l)
   
yy_test=[]
n=0
for i in y_test_ohe:
    x=ones((len_t[n],1))*i
    yy_test.append(x)
    n=n+1
yy_ts=[]
for i in yy_test:
    for n in i:
        yy_ts.append(n)



x_ct=tf.convert_to_tensor(xx_test)
y_c_t=tf.convert_to_tensor(yy_ts)

########


#network
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Activation, Dense, RepeatVector, Dropout, BatchNormalization
from tensorflow.keras import Sequential


#build model
inputs = keras.Input(shape=(1024))
dense_1=layers.Dense(512, activation='relu')(inputs)
batch=layers.BatchNormalization()(dense_1)
dropout=layers.Dropout(0.3)(batch)
dense_2=layers.Dense(512, activation='relu')(dropout)
dense_3=layers.Dense(256,activation='relu')(dense_2)
output=layers.Dense(200, activation='sigmoid')(dense_3)
model = keras.Model(inputs=inputs, outputs=output, name="Clssfr")
model.summary()


opt=tf.keras.optimizers.Adam(learning_rate=0.00001)   
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
history = model.fit(x_c, y_c, batch_size=64, epochs=30, validation_data=(x_c_v,y_c_v))


pred=[]
for i in X_test:
    y_pred=model.predict(i)
    pred.append(y_pred)


ind=[]
words_pred=[]
ind_max=0
for p in pred:
    ind=[]
    for i in range(0, len(p)):
        ind_m=np.argmax(p[i])
        ind.append(ind_m)
    frame_unique=np.unique(ind)
    counts = [ind.count(i) for i in frame_unique]
    idx_frame = np.argmax(counts)
    idx_max =  frame_unique[idx_frame]
    word =vocabulary[idx_max]
    words_pred.append(word)
    
words_predicted=pd.DataFrame(words_pred, columns=['words_predicted'])
words_predicted.to_csv('words_predicted_classifier.csv')
############################################################################################

#average_precision

words_predicted=pd.read_csv('words_predicted_classifier.csv')

w_p=words_predicted['words_predicted']

#create list of labels
#test
y_test_list=[]
w=[]
word=''
for i in y_test:
    for n in i:
        if n!=',':
            word=word+n
        else:
            w.append(word)
            word=''
    w.append(word)
    y_test_list.append(w)
    word=''
    w=[]


y_test_oh=[]
array=np.zeros([200])
ar=np.zeros([200])

for i in y_test_list:
    for n in i:
      for j in range(len(vocabulary)):
           ar=np.zeros([200])
           if n==vocabulary[j]:
                ar[j]=1
                array=array+ar
           else:
                j=j+1
    y_test_oh.append(array)
    array=np.zeros([200])
    ar=np.zeros([200])
    
#convert predicted words in vectors(0 and 1)
y_test_pred_oh=[]
array_=np.zeros([200])
arr=np.zeros([200])
for i in w_p:
    for n in range(len(voc)):
        if i==vocabulary[n]:
            array_[n]=1.0
            arr=arr+array_
        else:n=n+1
    y_test_pred_oh.append(arr)
    array_=np.zeros([200])
    arr=np.zeros([200])


y_t=np.array(y_test_oh)   
y_t_p=np.array(y_test_pred_oh)

from sklearn import metrics
from sklearn.metrics import average_precision_score
a_precision_pred_mean = []
for i in range(len(vocabulary)):
    a_pre = metrics.average_precision_score(y_t[:,i], y_t_p[:,i])
    print("For  {} average precision score:{:.2f}".format(vocabulary[i], a_pre))
    a_precision_pred_mean.append(a_pre)

df_a_precision_pred_mean = pd.DataFrame(vocabulary, columns=['vocabulary'])
df_a_precision_pred_mean['vocabulary']=vocabulary
df_a_precision_pred_mean['Average_precision_classifier'] = a_precision_pred_mean

df_a_precision_pred_mean.to_csv('Average_precision_classifier.csv')



