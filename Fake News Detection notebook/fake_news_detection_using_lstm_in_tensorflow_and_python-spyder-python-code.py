# -*- coding: utf-8 -*-
"""Fake News Detection using LSTM in Tensorflow and Python.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xczXvd0Z06X-UVQ4VPS9yk66miIeNsJ1

## 1. Importing all the Libraries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

"""## 2. Exploring Fake News"""

fake = pd.read_csv('data/Fake.csv')

fake.head()

fake.columns

fake['subject'].value_counts()

fake['subject'].unique()

plt.figure(figsize=(10,6))
sns.countplot(x='subject',data=fake)

"""WordCloud for Fake news"""

type(fake['text'].tolist())

text = ' '.join(fake['text'].tolist())

wordcloud = WordCloud(width=1920, height=1080).generate(text)
fig = plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

"""From above word cloud we can see the words which are mostly repeated in context of fake news
Most repeated words are United States America , now , Donald Trump since the dataset used for analysis is from US

### 3. Exploring Fake News
"""

real = pd.read_csv('data/True.csv')

real.head()

real.columns

real['subject'].value_counts()

"""WordCloud for Real News"""

text = ' '.join(real['text'].tolist())

wordcloud = WordCloud(width=1920, height=1080).generate(text)
fig = plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

"""From above WordCloud we can see that most repeated words are Washington Reuters, Reuters President and more, from this we can say that publishers are most repeated in Real news"""

sns.countplot(x='subject',data=real)

real.isnull().sum()

"""## Difference in Text
Real news seem to have source of publication which is not present in fake news set
Looking at the Data:
- Most of text contains reuters information such as "WASHINGTON (Reuters)".
- Some text are tweets from Twitter
- Few text do not contain any publication info

## Cleaning Data
Removing Reuters or Twitter Tweet information from the text
- Text can be splitted only once at "-" which is always present after mentioning source of publication, this gives us publication part and text part
- If we do not get text part, this means publication details was't given for the record
- The Twitter tweets always have same source, a long text of max 256 characters.
"""

real.sample(5)

# Making a new columns for unKnown_publishers i.e. not having any publisher information by splitting the real news by '-'
unknown_publishers = []
for index, row in enumerate(real.text.values):
    try:
        # Publisher is mentioned in record and taking zeroth column
        record = row.split('-',maxsplit=1)
        # printing the record with index 1 if not present then it will give error
        record[1]
        # Assertions are simply boolean expressions that check if the conditions return true or not. 
        # If it is true, the program does nothing and moves to the next line of code. However, if it's false, the program stops and throws an error.
        assert(len(record[0])<120)
    except:
        unknown_publishers.append(index)

len(unknown_publishers)

real.iloc[unknown_publishers].text

# Since row 8970 has no data so dropping it
real.iloc[8970]
real = real.drop(8970, axis=0)

real.shape

# From text column removing name of publisher and appending in publisher column
# Texts having no publisher then in then appending 'unknown' in publisher column
# Texts having publisher then in then appending 'publisher name' in publisher column

publisher = []
tmp_text = []

for index, row in enumerate(real.text.values):
    if index in unknown_publishers:
        tmp_text.append(row)
        publisher.append('Unknown')
    else:
        record = row.split('-', maxsplit=1)
        publisher.append(record[0].strip())
        tmp_text.append(record[1].strip())

real['publisher'] = publisher
real['text'] = tmp_text

real.head()

real.shape

# For fake news in some rows the text data is empty and only title is present means the Title is the only content
empty_fake_index = [index for index,text in enumerate(fake.text.tolist()) if str(text).strip() == ""]

len(empty_fake_index)

# Merging the text and title columns in one
real['text'] = real['title'] + " " + real['text']
fake['text'] = fake['title'] + " " + fake['text']

# Converting the data to lower case
real['text'] = real['text'].apply(lambda x:str(x).lower())
fake['text'] = fake['text'].apply(lambda x:str(x).lower())

"""## Preprocessing Text"""

# Since Fake news classifier is Supervised Machine Learning Problem so adding labels 
real['class'] = 1
fake['class'] = 0

"""Combining both Fake and Real Dataset"""

real.columns

real = real[['text','class']]

fake = fake[['text','class']]

data = real.append(fake, ignore_index=True)

data.shape

data.sample(5)

"""## Vectorization - Word2Vec

Word2Vec is one of the most popular technique to learn word embeddings using shallow neural network
Word embedding is the most popular representation of document vocabulary. It is capable of capturing context of word in a document semantic and syntactic, relation with other words,etc.

<img src="https://th.bing.com/th/id/R.85dad8627ae6845b62f5bb965c291b19?rik=DK3U9M0C6y4weg&riu=http%3a%2f%2fjalammar.github.io%2fimages%2fword2vec%2fword2vec.png&ehk=l2HKjP2OyoOZzn3PanqJIxSM5nG7sgwwvQ6R702QxvE%3d&risl=&pid=ImgRaw&r=0" width='50%'>

<img src="https://www.samyzaf.com/ML/nlp/word2vec2.png" width='50%'>

# Using Gensim Library for word2vec
"""

# !pip install gensim

import gensim # for converting words to vector
from gensim.models import Word2Vec

y = data['class'].values

# Converting all the texts to list 
X = [d.split() for d in data['text'].tolist()]

w2v_model = Word2Vec(sentences=X, window=10, min_count=1)

len(w2v_model.wv)

similar = w2v_model.wv.most_similar('india')

similar

w2v_model.wv['india']

"""# 4.Preprocessing the data """

# Tokenizer will remove special characters and converts the characters to lower case
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)

X = tokenizer.texts_to_sequences(X)

tokenizer.word_index

"""Analysis the text data since most of news range between 600 to 1400 so taking maxelngth of news as 1050"""

plt.hist([len(x) for x in X],bins=700)
plt.show()

"""Since we can see that most of the data or text are less than 1000 words so ignore data have more than 1000 words data"""

nos = np.array([len(x) for x in X])
len(nos[nos>1000])

maxlen = 1000
X = pad_sequences(X,maxlen=maxlen)

X[0]

# Adding +1 for unknown words
vocab_size = len(tokenizer.word_index) + 1

vocab_size

DIM = 100
vocab = tokenizer.word_index
def get_weight_matrix(model):
    weight_matrix = np.zeros((vocab_size, DIM))
    
    for word,i in vocab.items():
        weight_matrix[i] = model.wv[word]
    
    return weight_matrix

embedding_vectors = get_weight_matrix(w2v_model)

embedding_vectors.shape

'''# 5. Creating model using word embedding and lSTM in Deep learning'''
model = Sequential()
model.add(Embedding(vocab_size, output_dim=DIM, weights=[embedding_vectors], input_length=maxlen, trainable=False))
model.add(LSTM(units=128))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# Check its architecture
model.summary()

'''Splitting in Train-Test'''
X_train, X_test, y_train, y_test = train_test_split(X,y)

'''Fitting the model on Train data i.e. Training the model on data using 10 epoches'''
model.fit(X_train, y_train, validation_split=0.3, epochs=10)

'''Accuracy on Test Dataset'''
y_pred = (model.predict(X_test) >= 0.5).astype(int)

accuracy_score(y_test, y_pred)
'''Printing confusion matrix ,precision and recall'''
print(classification_report(y_test, y_pred))

"""# 6. How to  classify on custom data by same model"""

x=['''The Minister said in the Rajya Sabha that the UK has informed that 62 files on Bose are already available on the websites of the National Archives and the British Library.MoS Muraleedharan was replying to a question on the governments efforts to seek cooperation relating to the controversy over Netaji's death.
The Russian Government had informed the government of India that they were unable to find any documents in the Russian archives pertaining to Netaji. "The Russian government said that additional investigations were made to find the documents, based on request from the Indian side," he said.
ALSO READ: PM Modi unveils hologram statue of Netaji Subhas Chandra Bose at India Gate
The Japanese government has declassified two files on Netaji. "These files are part of their Archives and are available in the public domain. The government of Japan has transferred these files to India and they are retained in the National Archives of India," the minister said.
Muraleedharan informed the government of Japan has also said that if there are any additional documents relevant to the matter, those would be declassified as per their policies after a prescribed time period and based on an internal review mechanism.''']

x = tokenizer.texts_to_sequences(x)
print(x)
x = pad_sequences(x,maxlen=1000)
prediction = model.predict(x)
print(prediction)
print((prediction >=0.5).astype(int))

# -------------------------------------------------
'''# 7. Saving the model'''
import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

# saving the model
model.save('saved_model/model_pooja_bhagat_fake_news_classifier')

# Check its architecture
new_model.summary()

# -------------------------------------------------
'''8. Saving the model in h5 file for deployment'''
import tensorflow as tf
new_model = tf.keras.models.load_model('save_model/model_pooja_bhagat')

# Check its architecture
new_model.summary()

# saving model in .h5 file for deployment
model.save('Fake_news_classifier_pooja.h5')

# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('my_model_pooja.h5')

# Show the model architecture
new_model.summary()

# -------------------------------------------------
"""# 9. How to  classify on custom data by deployed model on fake news"""
# fake news taken 
x = [''' Last year a mysterious shipment was caught smuggling Coronavirus from Canada. It was traced to Chinese agents working at a Canadian lab. Subsequent investigation by GreatGameIndia linked the agents to Chinese Biological Warfare Program from where the virus is suspected to have leaked causing the Wuhan Coronavirus outbreak.
Note: BuzzFeed Reporter Who Attacked GreatGameIndia’s Coronavirus Bioweapon Story, Fired For Plagiarism
The findings of this investigation has been corroborated by none other than the Bioweapons expert Dr. Francis Boyle who drafted the Biological Weapons Convention Act followed by many nations. The report has caused a major international controversy and is suppressed actively by a section of mainstream media. ''']
x = tokenizer.texts_to_sequences(x)
print(x)
x = pad_sequences(x,maxlen=1000)
print(x)
# Prediction by deployed model
val = new_model.predict(x)
# Prediction by model created
val = model.predict(x)
print(val)
print(( val>=0.5).astype(int))


# -------------------------------------------------------------------
"""# 10. How to  classify on custom data by deployed model on real news"""
# True news
x = [''' The uncertainty created by Russia's invasion of Ukraine and its impact on the global economy is piling more complexity onto the US central bank's already tough fight to contain rising prices. Soaring energy and food costs have pushed inflation in the world's largest economy to the highest pace in four decades, and the Federal Reserve is poised to raise the benchmark borrowing rate in March to put out the fire. But while the Russia-Ukraine conflict is driving oil prices even higher, it also threatens to undercut the economic recovery from the Covid-19 pandemic. "It just makes a time that was always going to be challenging all the more so," Erica Goshen, a former senior Fed official, told AFP. Fed policymakers will be "watching the data very carefully. It throws a few more considerations into the pot," said Goshen, senior economics advisor at Cornell University's School of Industrial and Labor Relations. Crude prices briefly topped $100 a barrel on Thursday after Russia launched its invasion, the first time it passed that benchmark since 2014. And wheat prices also could spike, as Ukraine is one of the top global exporters of the grain.''']

tokenizer = Tokenizer(num_words=1000)
tokenizer.texts_to_sequences(x)
print(x)

word_index = tokenizer.word_index
word_index

x = pad_sequences(x,maxlen=1000)
print(x)
# Prediction by deployed model
val = new_model.predict(x)
print(val)
# Prediction by model created
val = model.predict(x)
print(val)
print(( val>=0.5).astype(int))

