import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, RMSprop
import pickle
import numpy as np
import os
import string

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard

from tensorflow import keras
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
import pickle
from keras.models import Sequential, load_model

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
#os.environ["PATH"] += os.pathsep + 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin'
#os.environ["PATH"] += os.pathsep + 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/extras/CUPTI/lib64'
#os.environ["PATH"] += os.pathsep + 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/include'


#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# open already preprocessed txt file with read "r"
file = open("HarryPotter_Ready.txt", "r", encoding = "utf8")
lines = []

# append each line in file to new array lines
for i in file:
    lines.append(i)
    
#print("The First Line: ", lines[0])
#print("The Last Line: ", lines[-1])

# create empty string
data = ""

# add lines to string
for i in lines:
    data = ' '. join(lines)

# replace extra spaces and new lines so that string is a single line of words    
data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '')

data = data.split()
data = ' '.join(data)

# create tokenizer for testing
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

# saving the tokenizer for predict function.
pickle.dump(tokenizer, open('tokenizer2.pkl', 'wb'))


sequence_data = tokenizer.texts_to_sequences([data])[0]
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)

# create sequences of five words with the sixth word being the prediction word
sequences = []
for i in range(5, len(sequence_data)):
    words = sequence_data[i-5:i+1]
    sequences.append(words)
    
print("The Length of sequences are: ", len(sequences))
sequences = np.array(sequences)
sequences[:10]

X = []
y = []

# creating sequence and prediction training/testing lists
for i in sequences:
    X.append(i[0:5])
    y.append(i[5])
    
X = np.array(X)
y = np.array(y)

print("The Data is: ", X[:5])
print("The responses are: ", y[:5])

y = to_categorical(y, num_classes=vocab_size)
y[:5]

# create LSTM model with 1 input layer, three hidden layers, and an output layer
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=5))
model.add(LSTM(1000, return_sequences=True))
model.add(LSTM(1000))
model.add(Dense(1000, activation="relu"))
model.add(Dense(vocab_size, activation="softmax"))

# display model summary
model.summary()

# plot the model tp png file
plot_model(model, to_file='model.png', show_layer_names=True)

# save the model
checkpoint = ModelCheckpoint("nextword2.h5", monitor='loss', verbose=1,
    save_best_only=True, mode='auto')

reduce = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001, verbose = 1)

logdir='logsnextword2'
tensorboard_Visualization = TensorBoard(log_dir=logdir)

optimizer1 = RMSprop(learning_rate=0.01)
optimizer2 = Adam(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer2, metrics=['accuracy'])

history = model.fit(X, y, epochs=40, batch_size=64, validation_split=0.20, callbacks=[checkpoint, reduce, tensorboard_Visualization])

# save the model and pickle it to pickle file
model.save("nextword2.h5")
pickle.dump(history.history, open("history2.p", "wb"))

model = load_model('nextword2.h5')
history = pickle.load(open('history2.p', 'rb'))

# plot the accuracy and loss graphs
plt.subplot(211)
plt.title('model accuracy')
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


plt.subplot(212)
plt.title('model loss')
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.tight_layout()

# https://stackoverflow.com/questions/26649716/how-to-show-pil-image-in-ipython-notebook
# tensorboard --logdir="./logsnextword1"
# http://DESKTOP-U3TSCVT:6006/

from IPython.display import Image 
#pil_img = Image(filename='model.png')

# display(pil_img)
#pil_img