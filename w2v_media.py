import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import keras
import pickle
import gensim
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords



# Load word2vec model (trained on an enormous Google corpus)
model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/carme/Downloads/GoogleNews-vectors-negative300.bin.gz', binary = True) 


model.vector_size

##############################################################################################################################################

path='C:/Users/carme/Desktop/FSD50K dataset/'

base_data_path=path+'unsplit/FSD50K.dev_audio/'




#SAVE EMBEDDINGS AND LABELS IN CSV
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
##########################################################################################################

#save y_train and embeddings 
data={'y_train':y_train, 
      'train_embeddings': X_train}

with open('train_data.json', 'wb') as fp:
    pickle.dump(data, fp)

###########################################################################################################
# val data
filelist_val = val_df['filename']

X_val=[]

for filename in filelist_val:
    s=np.load('C:/Users/carme/Desktop/FSD50K dataset/unsplit/SaveEmbeddings/validation/'+str(filename)+'.npy')
    X_val.append(s)

y_val=np.array(val_df['label'])
########################################################################################################
#save y_val and embeddings

data={'y_val':y_val, 
      'val_embeddings': X_val}

with open('val_data.json', 'wb') as fp:
    pickle.dump(data, fp)

#######################################################################################################

#test
filelist_test = test_df['filename']
X_test=[]

for filename in filelist_test:
    s=np.load('C:/Users/carme/Desktop/FSD50K dataset/unsplit/SaveEmbeddings/test/'+str(filename)+'.npy')
    X_test.append(s)

y_test=np.array(test_df['label'])
########################################################################################################

#save y_test and embeddings
data={'y_test':y_test, 
      'test_embeddings': X_test}

with open('test_data.json', 'wb') as fp:
    pickle.dump(data, fp)

##############################################################################################################

#word2vec
#create w2v of the vocabulary to associate them to train, val and test labels
voc=pd.read_csv('voc.csv')
label=voc['label']

vocabulary=[]
for i in label:

    # Tokenize the string into words
    tokens = word_tokenize(i)

    tok=[]
    w=''
    for i in tokens:
        for n in i:
         if n!=',' and n!='_':
            w=w+n
         else:
            tok.append(w)
            w=''
    tok.append(w)
    # Remove non-alphabetic tokens
    words = [word.lower() for word in tok if word.isalpha()]

    # Filter out stopwords
    stop_words = set(stopwords.words('english'))

    
    words = [word for word in words if not word in stop_words]
    vector_list=[]
    for i in words:
        if i in model:
            l=model[i]
            vector_list.append(l)
     
    word_vec_zip = zip(words, vector_list)

    
    word_vec_dict = dict(word_vec_zip)
    df = pd.DataFrame.from_dict(word_vec_dict, orient='index')
    vocabulary.append(df)

#mean value of the words for each label in the vocabulary
s=0
voc_mean=[]
el_to_delete=[]
v=0
for i in range(0,len(vocabulary)):
    l=len(vocabulary[i])
    if l>0:
        for n in range(0,len(vocabulary[i])):
                s=s+vocabulary[i].iloc[n]
        v=s/l
        voc_mean.append(v)
        s=0
        v=0
    else:
        el_to_delete.append(i)
        i=i+1

labels_list=list(label)
for i in el_to_delete:
    del labels_list[i]


##########################################################################################
   
#save vocabulary word2vec
w2v={'voc_label':labels_list,
     'w2v_voc':voc_mean}

with open('vocabulary_w2v.json', 'wb') as fp:
    pickle.dump(w2v, fp)
#########################################################################################################








#load embeddings and labels
#train

with open('train_data.json', 'rb') as fp:
    train = pickle.load(fp)
y_train=train['y_train']
X_train=train['train_embeddings']


#val
with open('val_data.json', 'rb') as fp:
   val = pickle.load(fp)
y_val=val['y_val']
X_val=val['val_embeddings']

#test
with open('test_data.json', 'rb') as fp:
    test = pickle.load(fp)
y_test=test['y_test']
X_test=test['test_embeddings']



    
##########################################################################################
    
#load vocabulary_w2vec
with open('vocabulary_w2v.json', 'rb') as fp:
    vocabulary_w2v= pickle.load(fp)

voc_w2v=vocabulary_w2v['w2v_voc'] #mean word embeddings 
voc=vocabulary_w2v['voc_label'] #corresponding labels

##############################################################################################
#PCA
from sklearn.decomposition import PCA
def ALGO(X_train, components_removed = 2):
    
    
    # PCA to get Top Components
    pca =  PCA(n_components = X_train.shape[1])
    X_train = X_train - np.mean(X_train)
    X_fit = pca.fit_transform(X_train)
    U1 = pca.components_
    
    z = []
    
    # Removing Projections on Top Components
    for i, x in enumerate(X_train):
    	for u in U1[0:components_removed]:        
            	x = x - np.dot(u.transpose(),x) * u 
    	z.append(x)
    
    z = np.asarray(z)
    
    # PCA Dim Reduction
    pca =  PCA(n_components = 150)
    X_train = z - np.mean(z)
    X_new_final = pca.fit_transform(X_train)
    
    return X_new_final

#v=np.array(voc_w2v)
#PCA_voc=ALGO(v, components_removed=3)


n_components=15
pca = PCA(n_components)
PCA_voc = pca.fit_transform(voc_w2v)
PCA_voc= pd.DataFrame(PCA_voc)
PCA_voc.to_csv('PCA_voc.csv', index=False)

############################################################################################################
#load PCA
voc_w2v=np.array(pd.read_csv('PCA_voc.csv'))


#################################################################################################
#create lists of labels for train, val and test 
#train
y_train_list=[]
w=[]
word=''
for i in y_train:
    for n in i:
        if n!=',':
            word=word+n
        else:
            w.append(word)
            word=''
    w.append(word)
    y_train_list.append(w)
    word=''
    w=[]


#val
y_val_list=[]
w=[]
word=''
for i in y_val:
    for n in i:
        if n!=',':
            word=word+n
        else:
            w.append(word)
            word=''
    w.append(word)
    y_val_list.append(w)
    word=''
    w=[]
    

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

#########################################################################################################
#combine w2v voc labels to train, val, test labels and calculate the mean value 
#to obtain a single vector for every label
s=0
k=0
y_train_w2v=[]
y_train_list
for i in y_train_list:
    for j in i:
        for n in range(0, len(voc)):
            if j!=voc[n]:
                n=n+1
            else:
                s=s+voc_w2v[n]
                k=k+1
    mean=s/k
    y_train_w2v.append(mean) 
    s=0
    k=0


#val
s=0
k=0
y_val_w2v=[]

for i in y_val_list:
    for j in i:
        for n in range(0, len(voc)):
            if j!=voc[n]:
                n=n+1
            else:
                s=s+voc_w2v[n]
                k=k+1
    mean=s/k
    y_val_w2v.append(mean) 
    s=0
    k=0


#test
s=0
k=0
y_test_w2v=[]

for i in y_test_list:
    for j in i:
        for n in range(0, len(voc)):
            if j!=voc[n]:
                n=n+1
            else:
                s=s+voc_w2v[n]
                k=k+1
    mean=s/k
    y_test_w2v.append(mean) 
    s=0
    k=0
####################################################################################################

#save train labels and w2v
data={'labels':y_train, 
      'word2vec':y_train_w2v}

with open('train_w2v.json', 'wb') as fp:
    pickle.dump(data, fp)


#save val labels and w2v
data={'labels':y_val, 
      'word2vec':y_val_w2v}

with open('val_w2v.json', 'wb') as fp:
    pickle.dump(data, fp)


#save val labels and w2v
data={'labels':y_test, 
      'word2vec':y_test_w2v}

with open('test_w2v.json', 'wb') as fp:
    pickle.dump(data, fp)

##################################################################################################











#load w2v

#train

with open('train_w2v.json', 'rb') as fp:
    train_w2v= pickle.load(fp)

train_vectors=train_w2v['word2vec'] 
labels_train=train_w2v['labels'] 


#val

with open('val_w2v.json', 'rb') as fp:
    val_w2v= pickle.load(fp)

val_vectors=val_w2v['word2vec'] 
labels_val=val_w2v['labels'] 


#test
with open('test_w2v.json', 'rb') as fp:
    test_w2v= pickle.load(fp)

test_vectors=test_w2v['word2vec'] 
labels_test=test_w2v['labels'] 



##########################################################################################################
#create a list of vectors in which each vector is the mean values of frames for every single embedding
from numpy import ones

xx_t=[]
len_x=[]
x=0
for i in X_train:
    x=np.mean(i, axis=0)
    xx_t.append(x)
    
    
#convert xx_t and train_vectors (w2v) in tensors
x_c=tf.convert_to_tensor(xx_t)
y_c=tf.convert_to_tensor(train_vectors)




#validation
xx_v=[]
len_v=[]
x=0
for i in X_val:
    x=np.mean(i, axis=0)
    xx_v.append(x)
    
       
x_c_v=tf.convert_to_tensor(xx_v)
y_c_v=tf.convert_to_tensor(val_vectors)



#test      
xx_test=[]
len_t=[]
x=0

for i in X_test:
    x = np.mean(i, axis = 0)
    xx_test.append(x)
  

x_ct=tf.convert_to_tensor(xx_test)
#y_c_t=tf.convert_to_tensor(yy_ts)

###################################################################


#network
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Activation, Dense, RepeatVector, Dropout, BatchNormalization
from tensorflow.keras import Sequential

from keras import backend

#build model
inputs = keras.Input(shape=(1024))
dense_1=layers.Dense(512, activation='relu')(inputs)
batch=layers.BatchNormalization()(dense_1)
dropout=layers.Dropout(0.3)(batch)
dense_2=layers.Dense(256, activation='relu')(dropout)
output=layers.Dense(30, activation='linear')(dense_2)
model_MSE = keras.Model(inputs=inputs, outputs=output, name="Clssfr")
model_MSE.summary()

def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


opt=tf.keras.optimizers.Adam(learning_rate=0.00001)   
model_MSE.compile(loss='MSE', optimizer=opt, metrics=[rmse])
history = model_MSE.fit(x_c, y_c, batch_size=64, epochs=30, validation_data=(x_c_v,y_c_v))

plt.plot(history.history['rmse'])
plt.title('RMSE_PCA_15_30 epoche')
plt.show()
plt.plot(history.history['loss'])
plt.title('LOSS_PCA_15_30epoche')
plt.show()


y_pred=model_MSE.predict(x_ct)
 

###################################################################################################
#create a list of 198 elements for every y_pred vector 
#each list contains the cosine similarity values between y_pred vector and vocabulary labels
from sklearn.metrics.pairwise import cosine_similarity

words_predicted=[]
w2v=[]
for p in y_pred:  
    cos=[]
    cos=list(cosine_similarity(voc_w2v, p.reshape(1,-1)))
    max_=np.max(cos)
    ind_max=cos.index(max_)
    word =voc[ind_max]
    words_predicted.append(word)
    w2v_=voc_w2v[ind_max]
    w2v.append(w2v_)
 
 
words_predicted_df=pd.DataFrame(w2v)
words_predicted_df['words_predicted']=words_predicted
words_predicted_df.to_csv('words_predicted_30_PCA_mean_.csv')

#######################################################################

#similarity matrix
#load words predicted 
words_predicted=pd.read_csv('words_predicted_15_PCA_mean_.csv')
words_predicted=words_predicted['words_predicted']




import numpy as np 
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
    
#macro classes:human, animal, music,things and nature  
Audioset_subclasses=[['human_voice, whistling, respiratory_sound,human_locomotion, digestive, hands, heart_sound, heartbeat, otoacoustic_emission, human_group_action'],
                     ['domestic_animal, pets, livestock, farm_animals, working_animals, wild_animals'],
                     ['musical_instrument, music_genre, musical_concepts, music_role, music_mood'],
                     ['vehicle, engine, domestic_sounds, home_sounds, bell, alarm, mechanisms, tools, explosion, wood, glass, liquid, miscellaneous_sources, specific_impact_sounds'],
                     ['wind, thunderstorm, water, fire']]

#list of labels
audioset_list=[]
w=[]
word=''
for i in Audioset_subclasses:
    for j in i:
        for n in j:
         if n!=',':
            word=word+n
         else:
            w.append(word)
            word=''
    w.append(word)
    audioset_list.append(w)
    word=''
    w=[]


#lists of w2v
a_sc=[]
a_sc_=[]
for n in audioset_list:
  a_sc_=[]
  for i in n:
    # Tokenize the string into words
    tokens = word_tokenize(i)

    tok=[]
    w=''
    for i in tokens:
        for n in i:
         if n!=',' and n!='_':
            w=w+n
         else:
            tok.append(w)
            w=''
    tok.append(w)
    # Remove non-alphabetic tokens
    words = [word.lower() for word in tok if word.isalpha()]

    # Filter out stopwords
    stop_words = set(stopwords.words('english'))

    
    words = [word for word in words if not word in stop_words]
    vector_list=[]
    for i in words:
        if i in model:
            l=model[i]
            vector_list.append(l)
     
    word_vec_zip = zip(words, vector_list)

    
    word_vec_dict = dict(word_vec_zip)
    df = pd.DataFrame.from_dict(word_vec_dict, orient='index')
    a_sc_.append(vector_list)
  a_sc.append(a_sc_)


#mean values
           
lbl_a=[]
aa=np.zeros([300])
arr=np.zeros([300])
for i in a_sc:
    aa=np.zeros([300])
    for n in i:
        arr=np.zeros([300])
        for j in n:
            arr=arr+j
        a=arr/len(n)
        aa=aa+a
    aa_=aa/len(i)
    lbl_a.append(aa_)
##################################################################################
#lists of labels
ylist_true=[]
w=[]
word=''
for i in labels_test:
    for n in i:
        if n!=',':
            word=word+n
        else:
            w.append(word)
            word=''
    w.append(word)
    ylist_true.append(w)
    word=''
    w=[]

#labels w2v
lbl_t=[]
lbl_t_=[]
for n in ylist_true:
  lbl_t_=[]
  for i in n:
    # Tokenize the string into words
    tokens = word_tokenize(i)

    tok=[]
    w=''
    for i in tokens:
        for n in i:
         if n!=',' and n!='_':
            w=w+n
         else:
            tok.append(w)
            w=''
    tok.append(w)
    # Remove non-alphabetic tokens
    words = [word.lower() for word in tok if word.isalpha()]

    # Filter out stopwords
    stop_words = set(stopwords.words('english'))

    
    words = [word for word in words if not word in stop_words]
    vector_list=[]
    for i in words:
        if i in model:
            l=model[i]
            vector_list.append(l)
        
    
    word_vec_zip = zip(words, vector_list)

    
    word_vec_dict = dict(word_vec_zip)
    
    df = pd.DataFrame.from_dict(word_vec_dict, orient='index')
    lbl_t_.append(vector_list)
  lbl_t.append(lbl_t_)

  
#mean values

lbl_true=[]
l_=[]
for i in lbl_t:
    l_=[]
    for n in i:
      if n!=[]:
        m_p=np.mean(n, axis=0)
        l_.append(m_p)
    m=np.mean(l_, axis=0)
    lbl_true.append(m)
    
 


############################################################################################################
#calculate cosine similarity between every single w2v and the Audioset macro classes 
#insert in a list the index of the maximum value

ind_true=[]
lista_=[]
lista_tot_true=[]
from scipy import spatial

for i in range(len(lbl_true)):
    lista_=[]
    for n in lbl_a:
        r=1-spatial.distance.cosine(n,lbl_true[i])
        lista_.append(r)
    m=np.argmax(lista_)
    ind_true.append(m)
    lista_tot_true.append(lista_)


#insert the word predicted in the macro classes list which belongs to
human=[]
animal=[]
music=[]
things=[]
nature=[]
ind_human=[]
ind_animal=[]
ind_music=[]
ind_things=[]
ind_nature=[]   
for i in range(len(ind_true)):
    
    if ind_true[i]==0:
        human.append(lbl_true[i])
        ind_human.append(i)
    if ind_true[i]==1:
        animal.append(lbl_true[i])
        ind_animal.append(i)
    if ind_true[i]==2:
        music.append(lbl_true[i])
        ind_music.append(i)
    if ind_true[i]==3:
        things.append(lbl_true[i])
        ind_things.append(i)
    if ind_true[i]==4:
        nature.append(lbl_true[i])
        ind_nature.append(i)
       
lista_true_=[human, music, animal, things, nature]
lista_true=[]
for i in lista_true_:
    for n in i:
        lista_true.append(n)
        
indici_=[ind_human, ind_music, ind_animal, ind_things, ind_nature]
indici=[]
for i in indici_:
    for n in i:
        indici.append(n)

################################################################################################
#list_pred
#list_pred=[]
#for i in ind_human[0:100]:
 #   list_pred.append(words_predicted[i])

#for i in ind_animal[0:100]:
 #   list_pred.append(words_predicted[i])

#for i in ind_music[0:100]:
 #   list_pred.append(words_predicted[i])

#for i in ind_things[0:100]:
 #   list_pred.append(words_predicted[i])

#for i in ind_nature[0:100]:
 #   list_pred.append(words_predicted[i])

lista_ordinata=[]
for i in indici:
    lista_ordinata.append(words_predicted[i])
    


#lists of words predicted
ypred=[]
w=[]
word=''
for i in lista_ordinata:
    for n in i:
        if n!=',':
            word=word+n
        else:
            w.append(word)
            word=''
    w.append(word)
    ypred.append(w)
    word=''
    w=[]


#w2v 
lbl_p=[]
lbl_p_=[]
for n in ypred:
  lbl_p_=[]
  for i in n:
    # Tokenize the string into words
    tokens = word_tokenize(i)

    tok=[]
    w=''
    for i in tokens:
        for n in i:
         if n!=',' and n!='_':
            w=w+n
         else:
            tok.append(w)
            w=''
    tok.append(w)
    # Remove non-alphabetic tokens
    words = [word.lower() for word in tok if word.isalpha()]

    # Filter out stopwords
    stop_words = set(stopwords.words('english'))

    
    words = [word for word in words if not word in stop_words]
    vector_list=[]
    for i in words:
        if i in model:
            l=model[i]
            vector_list.append(l)
        
    
    word_vec_zip = zip(words, vector_list)

    
    word_vec_dict = dict(word_vec_zip)
    
    df = pd.DataFrame.from_dict(word_vec_dict, orient='index')
    lbl_p_.append(vector_list)
  lbl_p.append(lbl_p_)
  
###############################################################################
#mean values

lbl_predicted=[]


for i in lbl_p:
    for n in i:
        if n!=[]:
           m_=np.mean(n, axis=0)
           lbl_predicted.append(m_)
    





#################################################################################################

#cosine similarity between every single w2v and the Audioset macro classes 
#insert in a list the index of the maximum value
ind_pred=[]
lista_p_=[]
lista_tot_pred=[]
from scipy import spatial

for i in range(len(lbl_predicted)):
    lista_p_=[]
    for n in lbl_a:
        r=1-spatial.distance.cosine(n,lbl_predicted[i])
        lista_p_.append(r)
    m=np.argmax(lista_p_)
    ind_pred.append(m)
    lista_tot_pred.append(lista_p_)



#insert the word predicted in the macro classes list which belongs to
human_1=[]
ind_human_1=[]
animal_1=[]
ind_animal_1=[]
music_1=[]
ind_music_1=[]
things_1=[]
ind_things_1=[]
nature_1=[]
ind_nature_1=[]



for i in range(len(ind_pred)):
    
    if ind_pred[i]==0:
        human_1.append(lbl_predicted[i])
        ind_human_1.append(i)
    if ind_pred[i]==1:
        animal_1.append(lbl_predicted[i])
        ind_animal_1.append(i)
    if ind_pred[i]==2:
        music_1.append(lbl_predicted[i])
        ind_music_1.append(i)
    if ind_pred[i]==3:
        things_1.append(lbl_predicted[i])
        ind_things_1.append(i)
    if ind_pred[i]==4:
        nature_1.append(lbl_predicted[i])
        ind_nature_1.append(i)
    
        
        
lista_pred_1=[human_1, music_1, animal_1, things_1, nature_1]
lista_predicted_1=[]
for i in lista_pred_1:
    for n in i:
        lista_predicted_1.append(n)

lista_ind_1=[ind_human_1, ind_music_1, ind_animal_1, ind_things_1, ind_nature_1]
lista_index_1=[]
for i in lista_ind_1:
    for n in i:
        lista_index_1.append(n)
#################################################################
###########################################################################################
#correlation matrix
def csm(A,B):
    
    B=B-B.mean(axis=1)[:,np.newaxis]
    A=A-A.mean(axis=1)[:,np.newaxis]
    num=np.dot(A,B.T)
    p1=np.sqrt(np.sum(A**2,axis=1))[:,np.newaxis]
    p2=np.sqrt(np.sum(B**2,axis=1))[np.newaxis,:]
    return num/(p1*p2)

cm_true=csm(np.array(lista_true), np.array(lista_true))
cm_pred=csm(np.array(lista_predicted_1), np.array(lista_predicted_1))

ticks=[len(human)/2,
       len(human),
       len(human)+(len(music))/2,
       len(human)+len(music),
       len(human)+len(music)+(len(animal))/2,
       len(human)+len(animal)+len(music),
       len(human)+len(animal)+len(music)+(len(things))/2,
       len(human)+len(animal)+len(music)+len(things),
       len(human)+len(animal)+len(music)+len(things)+(len(nature))/2]

ticks_label=['human',' ', 'music',' ', 'animal', ' ', 'things',' ','nature']

plt.imshow(cm_true, cmap='viridis')
plt.colorbar()
plt.title('correlation_matrix_y_true')
plt.xticks(ticks=ticks, labels=ticks_label, rotation=45)
plt.yticks(ticks=ticks, labels=ticks_label)
plt.grid()
plt.show()


ticks_pred=[len(human_1)/2,
            len(human_1),
            len(human_1)+(len(music_1))/2,
            len(human_1)+len(music_1),
            len(human_1)+len(music_1)+(len(animal_1))/2,
            len(human_1)+len(animal_1)+len(music_1),
            len(human_1)+len(animal_1)+len(music_1)+(len(things_1))/2,
            len(human_1)+len(animal_1)+len(music_1)+len(things_1),
            len(human_1)+len(animal_1)+len(music_1)+len(things_1)+(len(nature_1))/2]



plt.imshow(cm_pred, cmap='viridis')
plt.colorbar()
plt.title('correlation_matrix_y_pred')
plt.xticks(ticks=ticks_pred, labels=ticks_label, rotation=45)
plt.yticks(ticks=ticks_pred, labels=ticks_label)
plt.grid()
plt.show()



#########################################################################################
#save results of the cosine similarity between y_true and y_predicted
words_predicted_df=pd.read_csv('words_predicted_30_PCA_mean_.csv')


from scipy import spatial
results=[]
r=0

for i in range(len(words_predicted_df)):
   
    r=1-spatial.distance.cosine(test_vectors[i],words_predicted_df.iloc[i, 1:(words_predicted_df.shape[1]-1)])
    results.append(r)

result=pd.DataFrame(y_test)
result['words_predicted']=words_predicted_df['words_predicted']
result['results']=results
result.to_csv('results_cs_30_PCA_mean.csv')

#############################################################################
#one hot encoding y_test
y_test_oh=[]
array=np.zeros([198])
ar=np.zeros([198])

for i in y_test_list:
    for n in i:
      for j in range(len(voc)):
           ar=np.zeros([198])
           if n==voc[j]:
                ar[j]=1
                array=array+ar
           else:
                j=j+1
    y_test_oh.append(array)
    array=np.zeros([198])
    ar=np.zeros([198])


#one hot encoding y_pred       
y_test_pred=[]
array_=np.zeros([198])
arr=np.zeros([198])
for i in words_predicted_df['words_predicted']:
    for n in range(len(voc)):
        if i==voc[n]:
            array_[n]=1
            #arr=arr+array_
        else:n=n+1
    y_test_pred.append(array_)
    array_=np.zeros([198])

y_t=np.array(y_test_oh)   
y_t_p=np.array(y_test_pred)


#average prediction
from sklearn import metrics
from sklearn.metrics import average_precision_score
a_precision_pred_mean = []
for i in range(len(voc)):
    a_pre = metrics.average_precision_score(y_t[:,i], y_t_p[:,i])
    print("For  {} average precision score:{:.2f}".format(voc[i], a_pre))
    a_precision_pred_mean.append(a_pre)
df_a_precision_pred_mean = pd.DataFrame(voc, columns=['vocabulary'])
df_a_precision_pred_mean['Average_precision_30_PCA_mean'] = a_precision_pred_mean
df_a_precision_pred_mean.to_csv('Average_precision_30_PCA_mean.csv')

##################################################################################

#evaluation 
#LOAD results
cs_15_PCA=pd.read_csv('results_cs_15_PCA_30_epochs_.csv') 
cs_17_PCA=pd.read_csv('results_cs_17_PCA_30_epochs_.csv')
cs_20_PCA=pd.read_csv('results_cs_20_PCA_30_epochs_.csv') 
cs_30_PCA=pd.read_csv('results_cs_30_PCA_30_epochs_.csv')
cs_50_PCA=pd.read_csv('results_cs_50_PCA_30_epochs_.csv') 
cs_70_PCA=pd.read_csv('results_cs_70_PCA_30_epochs_.csv')
cs_90_PCA=pd.read_csv('results_cs_90_PCA_30_epochs_.csv')
cs_100_PCA=pd.read_csv('results_cs_100_PCA_30_epochs_.csv') 
cs_180_PCA=pd.read_csv('results_cs_180_PCA_30_epochs_.csv') 
cs_300_PCA=pd.read_csv('results_cs_300_PCA_30_epochs_.csv') 

#calculate max value, min value, mean value and the number of values higher than 0.5, between 0 and 0.5
#between -0.5 and 0 and lower than -0.5
s=0
for i in cs_15_PCA['results']:
   s=s+i
mean_15_PCA=s/len(cs_15_PCA['results'])
max_15_PCA=np.max(cs_15_PCA['results'])
min_15_PCA=np.min(cs_15_PCA['results'])

n_greater_zero5_15_PCA=0
for i in cs_15_PCA['results']:
    if i>0.5:
        n_greater_zero5_15_PCA=n_greater_zero5_15_PCA+1
n_between_zero_zero5_15_PCA=0
for i in cs_15_PCA['results']:
    if 0<i<0.5:
        n_between_zero_zero5_15_PCA=n_between_zero_zero5_15_PCA+1
n_between_minuszero5_zero_15_PCA=0
for i in cs_15_PCA['results']:
    if -0.5<i<0:
        n_between_minuszero5_zero_15_PCA=n_between_minuszero5_zero_15_PCA+1
n_less_minuszero5_15_PCA=len(cs_15_PCA)-n_between_minuszero5_zero_15_PCA-n_between_zero_zero5_15_PCA-n_greater_zero5_15_PCA

value_15_PCA=[mean_15_PCA, max_15_PCA, min_15_PCA, n_greater_zero5_15_PCA,n_between_zero_zero5_15_PCA,n_between_minuszero5_zero_15_PCA,n_less_minuszero5_15_PCA]




s=0
for i in cs_17_PCA['results']:
   s=s+i
mean_17_PCA=s/len(cs_17_PCA['results'])
max_17_PCA=np.max(cs_17_PCA['results'])
min_17_PCA=np.min(cs_17_PCA['results'])

n_greater_zero5_17_PCA=0
for i in cs_17_PCA['results']:
    if i>0.5:
        n_greater_zero5_17_PCA=n_greater_zero5_17_PCA+1
n_between_zero_zero5_17_PCA=0
for i in cs_17_PCA['results']:
    if 0<i<0.5:
        n_between_zero_zero5_17_PCA=n_between_zero_zero5_17_PCA+1
n_between_minuszero5_zero_17_PCA=0
for i in cs_17_PCA['results']:
    if -0.5<i<0:
        n_between_minuszero5_zero_17_PCA=n_between_minuszero5_zero_17_PCA+1
n_less_minuszero5_17_PCA=len(cs_17_PCA)-n_between_minuszero5_zero_17_PCA-n_between_zero_zero5_17_PCA-n_greater_zero5_17_PCA

value_17_PCA=[mean_17_PCA, max_17_PCA, min_17_PCA, n_greater_zero5_17_PCA,n_between_zero_zero5_17_PCA,n_between_minuszero5_zero_17_PCA,n_less_minuszero5_17_PCA]








s=0
for i in cs_20_PCA['results']:
   s=s+i
mean_20_PCA=s/len(cs_20_PCA['results'])
max_20_PCA=np.max(cs_20_PCA['results'])
min_20_PCA=np.min(cs_20_PCA['results'])

n_greater_zero5_20_PCA=0
for i in cs_20_PCA['results']:
    if i>0.5:
        n_greater_zero5_20_PCA=n_greater_zero5_20_PCA+1
n_between_zero_zero5_20_PCA=0
for i in cs_20_PCA['results']:
    if 0<i<0.5:
        n_between_zero_zero5_20_PCA=n_between_zero_zero5_20_PCA+1
n_between_minuszero5_zero_20_PCA=0
for i in cs_20_PCA['results']:
    if -0.5<i<0:
        n_between_minuszero5_zero_20_PCA=n_between_minuszero5_zero_20_PCA+1
n_less_minuszero5_20_PCA=len(cs_20_PCA)-n_between_minuszero5_zero_20_PCA-n_between_zero_zero5_20_PCA-n_greater_zero5_20_PCA

value_20_PCA=[mean_20_PCA, max_20_PCA, min_20_PCA, n_greater_zero5_20_PCA,n_between_zero_zero5_20_PCA, n_between_minuszero5_zero_20_PCA,n_less_minuszero5_20_PCA]


s=0
for i in cs_30_PCA['results']:
   s=s+i
mean_30_PCA=s/len(cs_30_PCA['results'])
max_30_PCA=np.max(cs_30_PCA['results'])
min_30_PCA=np.min(cs_30_PCA['results'])

n_greater_zero5_30_PCA=0
for i in cs_30_PCA['results']:
    if i>0.5:
        n_greater_zero5_30_PCA=n_greater_zero5_30_PCA+1
n_between_zero_zero5_30_PCA=0
for i in cs_30_PCA['results']:
    if 0<i<0.5:
        n_between_zero_zero5_30_PCA=n_between_zero_zero5_30_PCA+1
n_between_minuszero5_zero_30_PCA=0
for i in cs_30_PCA['results']:
    if -0.5<i<0:
        n_between_minuszero5_zero_30_PCA=n_between_minuszero5_zero_30_PCA+1
n_less_minuszero5_30_PCA=len(cs_30_PCA)-n_between_minuszero5_zero_30_PCA-n_between_zero_zero5_30_PCA-n_greater_zero5_30_PCA

value_30_PCA=[mean_30_PCA, max_30_PCA, min_30_PCA, n_greater_zero5_30_PCA,n_between_zero_zero5_30_PCA, n_between_minuszero5_zero_30_PCA,n_less_minuszero5_30_PCA]



s=0
for i in cs_50_PCA['results']:
   s=s+i
mean_50_PCA=s/len(cs_50_PCA['results'])
max_50_PCA=np.max(cs_50_PCA['results'])
min_50_PCA=np.min(cs_50_PCA['results'])

n_greater_zero5_50_PCA=0
for i in cs_50_PCA['results']:
    if i>0.5:
        n_greater_zero5_50_PCA=n_greater_zero5_50_PCA+1
n_between_zero_zero5_50_PCA=0
for i in cs_50_PCA['results']:
    if 0<i<0.5:
        n_between_zero_zero5_50_PCA=n_between_zero_zero5_50_PCA+1
n_between_minuszero5_zero_50_PCA=0
for i in cs_50_PCA['results']:
    if -0.5<i<0:
        n_between_minuszero5_zero_50_PCA=n_between_minuszero5_zero_50_PCA+1
n_less_minuszero5_50_PCA=len(cs_50_PCA)-n_between_minuszero5_zero_50_PCA-n_between_zero_zero5_50_PCA-n_greater_zero5_50_PCA

value_50_PCA=[mean_50_PCA, max_50_PCA, min_50_PCA, n_greater_zero5_50_PCA,n_between_zero_zero5_50_PCA, n_between_minuszero5_zero_50_PCA,n_less_minuszero5_50_PCA]



s=0
for i in cs_70_PCA['results']:
   s=s+i
mean_70_PCA=s/len(cs_70_PCA['results'])
max_70_PCA=np.max(cs_70_PCA['results'])
min_70_PCA=np.min(cs_70_PCA['results'])

n_greater_zero5_70_PCA=0
for i in cs_70_PCA['results']:
    if i>0.5:
        n_greater_zero5_70_PCA=n_greater_zero5_70_PCA+1
n_between_zero_zero5_70_PCA=0
for i in cs_70_PCA['results']:
    if 0<i<0.5:
        n_between_zero_zero5_70_PCA=n_between_zero_zero5_70_PCA+1
n_between_minuszero5_zero_70_PCA=0
for i in cs_70_PCA['results']:
    if -0.5<i<0:
        n_between_minuszero5_zero_70_PCA=n_between_minuszero5_zero_70_PCA+1
n_less_minuszero5_70_PCA=len(cs_70_PCA)-n_between_minuszero5_zero_70_PCA-n_between_zero_zero5_70_PCA-n_greater_zero5_70_PCA

value_70_PCA=[mean_70_PCA, max_70_PCA, min_70_PCA, n_greater_zero5_70_PCA,n_between_zero_zero5_70_PCA, n_between_minuszero5_zero_70_PCA,n_less_minuszero5_70_PCA]



s=0
for i in cs_90_PCA['results']:
   s=s+i
mean_90_PCA=s/len(cs_90_PCA['results'])
max_90_PCA=np.max(cs_90_PCA['results'])
min_90_PCA=np.min(cs_90_PCA['results'])

n_greater_zero5_90_PCA=0
for i in cs_90_PCA['results']:
    if i>0.5:
        n_greater_zero5_90_PCA=n_greater_zero5_90_PCA+1
n_between_zero_zero5_90_PCA=0
for i in cs_90_PCA['results']:
    if 0<i<0.5:
        n_between_zero_zero5_90_PCA=n_between_zero_zero5_90_PCA+1
n_between_minuszero5_zero_90_PCA=0
for i in cs_90_PCA['results']:
    if -0.5<i<0:
        n_between_minuszero5_zero_90_PCA=n_between_minuszero5_zero_90_PCA+1
n_less_minuszero5_90_PCA=len(cs_90_PCA)-n_between_minuszero5_zero_90_PCA-n_between_zero_zero5_90_PCA-n_greater_zero5_90_PCA

value_90_PCA=[mean_90_PCA, max_90_PCA, min_90_PCA, n_greater_zero5_90_PCA,n_between_zero_zero5_90_PCA, n_between_minuszero5_zero_90_PCA,n_less_minuszero5_90_PCA]



s=0
for i in cs_100_PCA['results']:
   s=s+i
mean_100_PCA=s/len(cs_100_PCA['results'])
max_100_PCA=np.max(cs_100_PCA['results'])
min_100_PCA=np.min(cs_100_PCA['results'])

n_greater_zero5_100_PCA=0
for i in cs_100_PCA['results']:
    if i>0.5:
        n_greater_zero5_100_PCA=n_greater_zero5_100_PCA+1
n_between_zero_zero5_100_PCA=0
for i in cs_100_PCA['results']:
    if 0<i<0.5:
        n_between_zero_zero5_100_PCA=n_between_zero_zero5_100_PCA+1
n_between_minuszero5_zero_100_PCA=0
for i in cs_100_PCA['results']:
    if -0.5<i<0:
        n_between_minuszero5_zero_100_PCA=n_between_minuszero5_zero_100_PCA+1

n_less_minuszero5_100_PCA=len(cs_100_PCA)-n_between_minuszero5_zero_100_PCA-n_between_zero_zero5_100_PCA-n_greater_zero5_100_PCA

value_100_PCA=[mean_100_PCA, max_100_PCA, min_100_PCA, n_greater_zero5_100_PCA,n_between_zero_zero5_100_PCA, n_between_minuszero5_zero_100_PCA,n_less_minuszero5_100_PCA]


s=0  
for i in cs_180_PCA['results']:
   s=s+i
mean_180_PCA=s/len(cs_180_PCA['results'])
max_180_PCA=np.max(cs_180_PCA['results'])
min_180_PCA=np.min(cs_180_PCA['results'])

n_greater_zero5_180_PCA=0
for i in cs_180_PCA['results']:
    if i>0.5:
        n_greater_zero5_180_PCA=n_greater_zero5_180_PCA+1
n_between_zero_zero5_180_PCA=0
for i in cs_180_PCA['results']:
    if 0<i<0.5:
        n_between_zero_zero5_180_PCA=n_between_zero_zero5_180_PCA+1
n_between_minuszero5_zero_180_PCA=0
for i in cs_180_PCA['results']:
    if -0.5<i<0:
        n_between_minuszero5_zero_180_PCA=n_between_minuszero5_zero_180_PCA+1

n_less_minuszero5_180_PCA=len(cs_180_PCA)-n_between_minuszero5_zero_180_PCA-n_between_zero_zero5_180_PCA-n_greater_zero5_180_PCA

value_180_PCA=[mean_180_PCA, max_180_PCA, min_180_PCA, n_greater_zero5_180_PCA,n_between_zero_zero5_180_PCA, n_between_minuszero5_zero_180_PCA,n_less_minuszero5_180_PCA]




s=0  
for i in cs_300_PCA['results']:
   s=s+i
mean_300_PCA=s/len(cs_300_PCA['results'])
max_300_PCA=np.max(cs_300_PCA['results'])
min_300_PCA=np.min(cs_300_PCA['results'])

n_greater_zero5_300_PCA=0
for i in cs_300_PCA['results']:
    if i>0.5:
        n_greater_zero5_300_PCA=n_greater_zero5_300_PCA+1
n_between_zero_zero5_300_PCA=0
for i in cs_300_PCA['results']:
    if 0<i<0.5:
        n_between_zero_zero5_300_PCA=n_between_zero_zero5_300_PCA+1
n_between_minuszero5_zero_300_PCA=0
for i in cs_300_PCA['results']:
    if -0.5<i<0:
        n_between_minuszero5_zero_300_PCA=n_between_minuszero5_zero_300_PCA+1

n_less_minuszero5_300_PCA=len(cs_300_PCA)-n_between_minuszero5_zero_300_PCA-n_between_zero_zero5_300_PCA-n_greater_zero5_300_PCA

value_300_PCA=[mean_300_PCA, max_300_PCA, min_300_PCA, n_greater_zero5_300_PCA,n_between_zero_zero5_300_PCA, n_between_minuszero5_zero_300_PCA,n_less_minuszero5_300_PCA]




data = {'15_PCA':value_15_PCA ,'17_PCA':value_17_PCA, '20_PCA': value_20_PCA, '30_PCA': value_30_PCA, '50_PCA':value_50_PCA ,'70_PCA':value_70_PCA ,'90_PCA':value_90_PCA, '100_PCA':value_100_PCA,'180_PCA':value_180_PCA, '300_PCA':value_300_PCA}
df=pd.DataFrame.from_dict(data, orient='index', columns=['mean', 'max', 'min', 'cs>0.5', '0<cs<0.5', '-0.5<cs<0', 'cs<-0.5'])
df.to_csv('compare_PCA.csv')


    