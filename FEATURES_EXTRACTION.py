
import numpy as np
import soundfile as sf
import resampy
import params
import yamnet as yamnet_model
import tensorflow as tf
import scipy.io
import sys 
from tensorflow.keras import Model, layers
import pandas as pd   
import glob,os
import params as yamnet_params
import librosa
import tensorflow_io as tfio

if sys.platform == 'darwin':
    user_env = os.environ['PWD']
    project_path = os.path.join(user_env,'Desktop','ERASMUS2021_Traineeship','TraineeshipMariannaPasqualeCarmen')
    data_path =  os.path.join(user_env,'Desktop','ERASMUS2021_Traineeship','Data')

elif sys.platform == 'win32':
    user_env = os.environ['USERPROFILE']
    project_path = os.path.join(user_env,'Desktop', 'FSD50K dataset','unsplit','ERASMUS_2021','project_path','')
    data_path = os.path.join(user_env,'Desktop','FSD50K dataset','unsplit','ERASMUS_2021','Data','')

elif sys.platform == 'linux':
    user_env = '/workspace/notebooks/'
    project_path = os.path.join(user_env,'Natural-Sound-Analysis','')



def read_audio(fn):
    y, sr = librosa.load(fn, sr=16000)
    signal_len=len(y)
    T_s=1/sr
    if signal_len <= sr:
        time=int(np.ceil(signal_len*T_s))
    else:
        time = int(np.round(signal_len*T_s))

    # trim silence
    if 0 < len(y): # workaround: 0 length causes error
        y, _ = librosa.effects.trim(y, top_db=60) # trim, top_db=default(60)
   # make it unified length to conf.samples
    if len(y) > sr*time: # long enough
        y = y[0:sr*time]
    else: # pad blank
        padding = sr*time - len(y)    # add padding at both ends
        offset = padding // 2
        y = np.pad(y, (offset, sr*time - len(y) - offset), 'constant')
    return y,sr



checkpoint_path = 'yamnet.h5'

#load model and weights
params.PATCH_HOP_SECONDS=1
yamnet=yamnet_model.yamnet_frames_model(params)
yamnet.load_weights(checkpoint_path)

#for a look of all the architecture compute yamnet.summary()
yamnet.summary()



#VALIDATION 
#development audio
path='C:/Users/carme/Desktop/FSD50K dataset/unsplit/FSD50K.dev_audio/'

df_val=pd.read_csv('val.csv')
print(df_val)


df_val['file']=df_val['fname'].apply(str)+'.wav'
idx,  files= df_val.index.values, df_val.file.values
print(files)

yamnet.summary()
embedding=[]
features=[]

n=0
i=0
fname=[]
for i in files:
    i = i[:-4]
    fname.append(i)


for f in files:
   
    wav_file=path+f
    wave, sample_rate = read_audio(wav_file)
        
    # create a specific model that takes as input the input of yamnet, and outputs the layers
    extractor = Model(inputs=yamnet.input, outputs=yamnet.get_layer('global_average_pooling2d_4').output)
    
   
    features = extractor.predict(np.reshape(wave, [1, -1]), steps=1)
    print(features)
    
    #list of lists
    embedding.append(features) 
    np.save('C:/Users/carme/Desktop/FSD50K dataset/unsplit/SaveEmbeddings/validation/'+str(fname[n]), features)
    n+= 1

np.save('C:/Users/carme/Desktop/FSD50K dataset/unsplit/feature/'+str('emb_val'), embedding)


#TRAINING
#development audio
path='C:/Users/carme/Desktop/FSD50K dataset/unsplit/FSD50K.dev_audio/'

df_train=pd.read_csv('train.csv')
print(df_train)



df_train['file']=df_train['fname'].apply(str)+'.wav'
idx,  files= df_train.index.values, df_train.file.values
print(files)

yamnet.summary()
embedding=[]
features=[]

i=0
fname=[]
for i in files:
    i = i[:-4]
    fname.append(i)


for f in files:
   
    wav_file=path+f
    wave, sample_rate = read_audio(wav_file)
        
    # create a specific model that takes as input the input of yamnet, and outputs the layers
    extractor = Model(inputs=yamnet.input, outputs=yamnet.get_layer('global_average_pooling2d_3').output)
    
   
    features = extractor.predict(np.reshape(wave, [1, -1]), steps=1)
    print(features)
    
    #list of lists
    embedding.append(features) 
    np.save('C:/Users/carme/Desktop/FSD50K dataset/unsplit/SaveEmbeddings/train/'+str(fname[n]), features)
    n+= 1

np.save('C:/Users/carme/Desktop/FSD50K dataset/unsplit/feature/'+str('emb_train'), embedding)



#TEST
#evaluation audio
path='C:/Users/carme/Desktop/FSD50K dataset/eval/'

df_test=pd.read_csv('eval.csv')
print(df_test)


df_test['file']=df_test['fname'].apply(str)+'.wav'
idx,  files= df_test.index.values, df_test.file.values
print(files)

yamnet.summary()
embedding=[]
features=[]


i=0
fname=[]
for i in files:
    i = i[:-4]
    fname.append(i)


for f in files:
   
    wav_file=path+f
    wave, sample_rate = read_audio(wav_file)
        
    # create a specific model that takes as input the input of yamnet, and outputs the layers
    extractor = Model(inputs=yamnet.input, outputs=yamnet.get_layer('global_average_pooling2d_3').output)
    
   
    features = extractor.predict(np.reshape(wave, [1, -1]), steps=1)
    print(features)
    
    #list of lists
    embedding.append(features) 
    np.save('C:/Users/carme/Desktop/FSD50K dataset/unsplit/SaveEmbeddings/test/'+str(fname[n]), features)
    n+= 1

np.save('C:/Users/carme/Desktop/FSD50K dataset/unsplit/feature/'+str('emb_test'), embedding)


