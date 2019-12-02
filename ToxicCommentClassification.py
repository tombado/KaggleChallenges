#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import numpy as np
import re
import collections
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras import models
from keras import layers

data = pd.read_csv("Kaggletoxic/train.csv") 
data = data
#data = data.drop(["severe_toxic","obscene","threat","insult", "identity_hate"],axis=1)

def countCaps(string):
    
    return sum(1 for c in string if c.isupper())/len(string)
def countExcl(string):
    return sum(1 for c in string if c == "!" or c == "?" or c == "@")/len(string)



data["Excl"] = data["comment_text"].apply(lambda x: countExcl(x))

data["CAPS"] = data["comment_text"].apply(lambda x: countCaps(x))

data["text"] = data["comment_text"].apply(lambda x: x.split())

import string
table = str.maketrans('', '', string.punctuation)

data["text"] = data["text"].apply(lambda x: [w.translate(table) for w in x])
data["text"] = data["text"].apply(lambda x: [w.lower() for w in x])





stopwords = ['i', "im", 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
def removeStopWords(wordList):
    listCopy = []
    for word in wordList:
        if word not in stopwords:
            if word.isalpha():
                
                listCopy.append(word)
            
    return listCopy

data["text"] = data["text"].apply(lambda x: removeStopWords(x))

testdata = pd.read_csv("Kaggletoxic/test.csv")
testdata["Excl"] = testdata["comment_text"].apply(lambda x: countExcl(x))
testdata["CAPS"] = testdata["comment_text"].apply(lambda x: countCaps(x))
testdata["text"] = testdata["comment_text"].apply(lambda x: x.split())
testdata["text"] = testdata["text"].apply(lambda x: [w.translate(table) for w in x])
testdata["text"] = testdata["text"].apply(lambda x: [w.lower() for w in x])
testdata["text"] = testdata["text"].apply(lambda x: removeStopWords(x))


# In[46]:


labeldata = pd.read_csv("Kaggletoxic/test_labels.csv")

def makeString(wordList):
    

    return ' '.join(word for word in wordList)
testdata["finalText"] = testdata["text"].apply(lambda x: makeString(x))

data["finalText"] = data["text"].apply(lambda x: makeString(x))
docs = data["finalText"]
testlabels = labeldata[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]].values
extraFeature1 = data["CAPS"]
extraFeature2 = data["Excl"]
labels =data[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]].values
extratestFeature1 = testdata["CAPS"]
extratestFeature2 = testdata["Excl"] 
testdocs = testdata["finalText"]


# In[58]:





# In[47]:



# prepare tokenizer
t = Tokenizer(num_words=None, split=' ',)
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(docs)
encoded_testdocs = t.texts_to_sequences(testdocs)

#print(encoded_docs)
# pad documents to a max length of 4 words
max_length = 100
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
padded_testdocs = pad_sequences(encoded_testdocs, maxlen=max_length, padding='post')
#print(padded_docs)


# In[48]:


t.word_index


# In[49]:


# load the whole embedding into memory
embeddings_index = dict()
glove_file = 'Kaggletoxic/glove.twitter.27B/glove.twitter.27B.100d.txt'

f = open(glove_file,encoding = "utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


# In[50]:



# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(padded_docs, labels, test_size=0.2, random_state=37)

X_train_feature1, X_test_feature1, trashx, trashy = train_test_split(extraFeature1, labels, test_size=0.2, random_state=37)
X_train_feature2, X_test_feature2, trashx, trashy = train_test_split(extraFeature2, labels, test_size=0.2, random_state=37)


# In[ ]:





# In[52]:


from keras import Sequential
from keras.layers import Dense, Activation, Flatten, Embedding

# define model
#model = Sequential()
#e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=100, trainable=False)
#model.add(e)
##model.add(Flatten())
#model.add(Dense(1, activation='sigmoid'))
# compile the model
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
#print(model.summary())
# fit the model
#model.fit(X_train, y_train, epochs=10, verbose=1,validation_data=(X_test, y_test))
# evaluate the model
#loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
#print('Test loss:', loss)
#print('Test accuracy:', accuracy)


# In[53]:


from keras.models import Model
import keras
from keras.layers import concatenate


# In[54]:


from keras.layers import Input, Embedding, LSTM, Dense, Dropout, GRU
from keras.models import Model
import numpy as np
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=100, trainable=False)

np.random.seed(0)  # Set a random seed for reproducibility
main_input = Input(shape=(100,), dtype='int32', name='main_input')
x = e(main_input)
drop = Dropout(0.5)(x)


lstm_out = LSTM(32,return_sequences=True)(drop)
gru = GRU(32)(lstm_out)
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(gru)

auxiliary_input1 = Input(shape=(1,), name='aux_input1')
auxiliary_input2 = Input(shape=(1,), name='aux_input2')


x = keras.layers.concatenate([gru, auxiliary_input1])
x = keras.layers.concatenate([x, auxiliary_input2])


main_output = Dense(6, activation='sigmoid', name='main_output')(x)
model = Model(inputs=[main_input, auxiliary_input1,auxiliary_input2], outputs=[main_output])
model.compile(optimizer='rmsprop', loss='binary_crossentropy',)

#checkpoint = ModelCheckpoint('model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')  


# In[96]:


model.fit([X_train, X_train_feature1,X_train_feature2], y_train ,epochs=7, batch_size=512,validation_split=0.1)


# In[118]:


pred = model.predict([padded_testdocs, extratestFeature1,extratestFeature2])


# In[117]:


import numpy as np
from sklearn import metrics
#y = np.array([1, 1, 2, 2])
#pred = np.array([0.1, 0.4, 0.35, 0.8])
list1 = []
list2 = []
list3 = []
list4 = []
list5 = []
list6 = []
listpred1 = []
listpred2 = []
listpred3 = []
listpred4 = []
listpred5 = []
listpred6 = []

for i in y_test:
    
    list1.append(i[0])
    
    list2.append(i[1])
    list3.append(i[2])
    list4.append(i[3])
    list5.append(i[4])
    list6.append(i[5])
    
for j in pred:
    
    listpred1.append(j[0])
    listpred2.append(j[1])
    listpred3.append(j[2])
    listpred4.append(j[3])
    listpred5.append(j[4])
    listpred6.append(j[5])

fpr, tpr, thresholds = metrics.roc_curve(list1, listpred1, pos_label=1)
metrics.auc(fpr, tpr)


# In[114]:


listpred1


# In[ ]:





# In[119]:


fuckthislist=[]
for i in pred:
    fuckthislist.append([i[0],i[1],i[2],i[3],i[4],i[5]])


# In[89]:


result = []
for i in fuckthislist:
    resultlist = np.where(np.array(i) > 0.5, 1, 0)
    result.append(resultlist)


# In[95]:





# In[120]:


testdata = pd.read_csv("Kaggletoxic/test.csv")


# In[121]:


testdata = testdata.drop(["comment_text"],axis=1)
df=pd.DataFrame(fuckthislist,columns=["toxic","severe_toxic","obscene","threat","insult","identity_hate"])
df


# In[122]:


result = pd.concat([testdata, df], axis=1, sort=False)


# In[123]:


result


# In[124]:


result.to_csv('Kaggletoxic/submission.csv', index=False)


# In[29]:


counter=0
for i in X_test:
    if i[99] != 0:
        counter+=1


# In[30]:


counter


# In[31]:


len(X_test)


# In[ ]:




