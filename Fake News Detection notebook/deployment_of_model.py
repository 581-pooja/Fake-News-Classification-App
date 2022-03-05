# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 11:52:12 2022

@author: Pooja Bhagat
"""
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

x = ['''The uncertainty created by Russia's invasion of Ukraine and its impact on the global economy is piling more complexity onto the US central bank's already tough fight to contain rising prices. Soaring energy and food costs have pushed inflation in the world's largest economy to the highest pace in four decades, and the Federal Reserve is poised to raise the benchmark borrowing rate in March to put out the fire. But while the Russia-Ukraine conflict is driving oil prices even higher, it also threatens to undercut the economic recovery from the Covid-19 pandemic. "It just makes a time that was always going to be challenging all the more so," Erica Goshen, a former senior Fed official, told AFP. Fed policymakers will be "watching the data very carefully. It throws a few more considerations into the pot," said Goshen, senior economics advisor at Cornell University's School of Industrial and Labor Relations. Crude prices briefly topped $100 a barrel on Thursday after Russia launched its invasion, the first time it passed that benchmark since 2014. And wheat prices also could spike, as Ukraine is one of the top global exporters of the grain. 
''']

tokenizer = Tokenizer(num_words = 1000, oov_token="<OOV>")
tokenizer.fit_on_texts(x)
word_index = tokenizer.word_index

print(word_index)

sequences = tokenizer.texts_to_sequences(x)
padded = pad_sequences(sequences, maxlen=1000)

print(sequences)
print(padded)

# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('my_model_pooja.h5')

# Show the model architecture
model.summary()

# predicting the news by deployed model
prediction = model.predict(padded)
print(prediction)