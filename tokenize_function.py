from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import pickle
import numpy as np

# Open the file specified by filename in utf8 format
file = open("HarryPotter_Ready.txt", "r", encoding='utf8')

data = ""
for line in file:
    data = ' '.join(line)

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

# saving the tokenizer for predict function.
pickle.dump(tokenizer, open('tokenizer1.pkl', 'wb'))

sequence_data = tokenizer.texts_to_sequences([data])[0]

vocab_size = len(tokenizer.word_index) + 1

sequences = []

for i in range(1, len(sequence_data)):
    words = sequence_data[i-1:i+1]
    sequences.append(words)

sequences = np.array(sequences)

X = []
y = []

for i in sequences:
    X.append(i[0])
    y.append(i[1])

X = np.array(X)
y = np.array(y)

y = to_categorical(y, num_classes=vocab_size)
