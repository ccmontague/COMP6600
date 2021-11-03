#from keras.preprocessing.text import Tokenizer
#import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
#import heapq
import numpy as np
#import re
import matplotlib.pyplot as plt
#from keras.utils import to_categorical
#from doc3 import training_doc3
from keras.models import Sequential, load_model
##from keras.layers import LSTM
#from keras.layers import Embedding
##from keras.layers.core import Dense, Activation
##from keras.optimizers import RMSprop
#model = load_model("mymodel.h5")
#from keras.preprocessing.sequence import pad_sequences
##import pickle

training_data = 'HarryPotter.txt'
text = open(training_data).read().lower()
print('length of the corpus is: :', len(text))

tokenizer = RegexpTokenizer(r'w+')
words = tokenizer.tokenize(text)

unique_words = np.unique(words)
unique_word_index = dict((c, i) for i, c in enumerate(unique_words))


LENGTH_WORD = 5
next_words = []
prev_words = []
for j in range(len(words) - LENGTH_WORD):
     prev_words.append(words[j:j + LENGTH_WORD])
     next_words.append(words[j + LENGTH_WORD])
print(prev_words[0])
print(next_words[0])


X = np.zeros((len(prev_words), LENGTH_WORD, len(unique_words)), dtype=bool)
Y = np.zeros((len(next_words), len(unique_words)), dtype=bool)
for i, each_words in enumerate(prev_words):
   for j, each_word in enumerate(each_words):
        X[i, j, unique_word_index[each_word]] = 1
   Y[i, unique_word_index[next_words[i]]] = 1

#model = Sequential()
#model.add(LSTM(128, input_shape=(WORD_LENGTH, len(unique_words))))
#model.add(Dense(len(unique_words)))
#model.add(Activation('softmax'))




#cleaned = re.sub(r'\W+', ' ', training_data).lower()
#tokens = word_tokenize(cleaned)
#train_len = 3+1
#text_sequences = []
#for i in range(train_len,len(tokens)):
#    seq = tokens[i-train_len:i]
#    text_sequences.append(seq)
#sequences = {}
#count = 1
#for i in range(len(tokens)):
#    if tokens[i] not in sequences:
#        sequences[tokens[i]] = count
#        count += 1
#tokenizer = Tokenizer()
#tokenizer.fit_on_texts(text_sequences)
#sequences = tokenizer.texts_to_sequences(text_sequences) 

##Collecting some information   
#vocabulary_size = len(tokenizer.word_counts)+1

#n_sequences = np.empty([len(sequences),train_len], dtype='int32')
#for i in range(len(sequences)):
#    n_sequences[i] = sequences[i]

#-------------------------------------------------------------------------------------------------------------------------

#train_inputs = n_sequences[:,:-1]
#train_targets = n_sequences[:,-1]
#train_targets = to_categorical(train_targets, num_classes=vocabulary_size)
#seq_len = train_inputs.shape[1]
#train_inputs.shape
##print(train_targets[0])

#-------------------------------------------------------------------------------------------------------------------------

#model = Sequential()
#model.add(Embedding(vocabulary_size, seq_len, input_length=seq_len))
#model.add(LSTM(50,return_sequences=True))
#model.add(LSTM(50))
#model.add(Dense(50,activation='relu'))
#model.add(Dense(vocabulary_size, activation='softmax'))
#print(model.summary())
## compile network
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.fit(train_inputs,train_targets,epochs=500,verbose=1)
#model.save("mymodel.h5")

#-------------------------------------------------------------------------------------------------------------------------

#input_text = input().strip().lower()
#encoded_text = tokenizer.texts_to_sequences([input_text])[0]
#pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
#print(encoded_text, pad_encoded)
#for i in (model.predict(pad_encoded)[0]).argsort()[-3:][::-1]:
#  pred_word = tokenizer.index_word[i]
#  print("Next word suggestion:",pred_word)