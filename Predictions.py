# Importing the Libraries

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from termcolor import colored

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def Predict_Next_Words(model, tokenizer, text):
    """
        In this function we are using the tokenizer and models trained
        and we are creating the sequence of the text entered and then
        using our model to predict and return the predicted word.
    
    """
    """
    for i in range(3):
        sequence = tokenizer.texts_to_sequences([text])[0]
        sequence = np.array(sequence)
        
        ##preds = model.predict_classes(sequence)
        preds = np.argmax(model.predict(sequence), axis=-1)
        #print(preds)
        predicted_word = ""
        
        for key, value in tokenizer.word_index.items():
            if value == preds:
                predicted_word = key
                break
        
        #print(predicted_word)
        return predicted_word 
    """

    sequence = tokenizer.texts_to_sequences([text])
    sequence = np.array(sequence)
    preds = np.argmax(model.predict(sequence))
    predicted_word = ""

    for key, value in tokenizer.word_index.items():
        if value == preds:
            predicted_word = key
            break

    print(predicted_word)
    return predicted_word

def process_test_data(N, Input_Filename):
    file = open(Input_Filename, "r", encoding='utf8')
    word = []
    sentence = []
    split_sentence = []
    i = 0

    # Traversing file line by line
    for line in file:
        # splits each line into words and appends the words into sentences
        for w in line.split():
            i = i + 1
            sentence.append(w)
            if i == N:
                i = 0
                #sentence.insert(N-1, '_')
                test_input = ' '.join(sentence[0:N-1])
                word.append(sentence[N-1])
                split_sentence.append(test_input)
                sentence = []

    return split_sentence, word

def rnn_algorithm(N, split_sentences, word):
    # Load the model and tokenizer
    model = load_model('nextword2.h5')
    tokenizer = pickle.load(open('tokenizer2.pkl', 'rb'))
    y = 0
    score = 0
    
    for line in split_sentences:
        original_line = line
        found_word = ""
        try:
            line = line.split(" ")
            line = line[-1]

            line = ''.join(line)
            found_word = Predict_Next_Words(model, tokenizer, line)
        except:
            continue
        
        if found_word == word[y]:
            score = score + 1
            print(original_line + " " + colored(found_word, 'green'))
        else:
            score = score
            print(original_line + " " + colored(found_word, 'red') +
                  '; ' + colored('Real Word: ' + word[y], 'blue'))
        y = y + 1
    score = (score/y)*100
    return score

if __name__ == "__main__":
    Input_Filename = "HarryPotter_Ready.txt"
    Num_Words = 5
    split_sentence, word = process_test_data(Num_Words+1, Input_Filename)
    
    score = rnn_algorithm(Num_Words, split_sentence, word)
    print('Score is ' + str(score) + '%')