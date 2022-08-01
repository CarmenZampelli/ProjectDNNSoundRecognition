import pandas as pd
import pickle
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
import matplotlib.pyplot as plt
#test
with open('test_data.json', 'rb') as fp:
    test = pickle.load(fp)
y_test=test['y_test']
X_test=test['test_embeddings']

dataframe=pd.read_csv('lista_predicted.csv')
labels_leg = ['human', 'music', 'animal', 'things', 'nature']
list_=[]
ind_=[]


for i in labels_leg:
  j=0
  for n in range(len(dataframe)):
    if i==str(dataframe['category'][n]) and j<1000:
        list_.append(dataframe.iloc[n, 0:300])
        ind_.append(dataframe['index'][n])
        j=j+1
        
      

labels=[]
for i in ind_:
    labels.append(y_test[i])

df=pd.DataFrame(list_, index = labels)


import seaborn as sns


#labels_leg = ['human', 'music', 'animal', 'things', 'nature']
perplexity=35
#df = pd.DataFrame(graph_data, index = dataframe['Event'])
# Initialize t-SNE
tsne = TSNE(n_components = 2, init = 'pca', random_state = 10, perplexity = 100)
tsne_df = tsne.fit_transform(df)
sns.set()
# Initialize figure
col=np.array([np.zeros(1000),
              np.zeros(1000)+1,
              np.zeros(1000)+2,
              np.zeros(1000)+3,
              np.zeros(1000)+4])
col=col.reshape(5000,1)     
labels_leg = ['human', 'music', 'animal', 'things', 'nature']
fig, ax = plt.subplots(figsize = (30, 30))
ax = sns.scatterplot(tsne_df[:, 0], tsne_df[:, 1], alpha = 1, hue = np.squeeze(col), legend = labels_leg, palette=['green','orange','brown','dodgerblue','red'])
handles, labels  =  ax.get_legend_handles_labels()
#plt.title("Scatter plot MSOS of layer {0} and model: {1}".format(j, m))
plt.title('t-SNE')
ax.legend(handles, labels_leg ,
                     loc="best", title="Classes")
# Import adjustText, initialize list of texts
from adjustText import adjust_text
texts = []
words_to_plot = list(np.arange(0, 500, 30))
# Append words to list
for word in words_to_plot:
 texts.append(plt.text(tsne_df[word, 0], tsne_df[word, 1], df.index[word], fontsize = 14))
            
# Plot text using adjust_text (because overlapping text is hard to read)
adjust_text(texts, force_points = 0.4, force_text = 0.4, 
            expand_points = (2,1), expand_text = (1,2),
            arrowprops = dict(arrowstyle = "-", color = 'black', lw = 0.5))



plt.show()
