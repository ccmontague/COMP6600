import random
from nltk import data
import numpy as np
import pandas as panda
from nltk.tag import pos_tag
#from termcolor import colored

# Tag Parts of Speech using pre-processed Harry Potter


def tag_parts_of_speech(filename):
    with open(filename, 'r') as input_text:
        input_string = input_text.read()
        words = input_string.split()
        tagged_text = pos_tag(words, tagset='universal')
        # Print a sample of 10 tagged words
        # Tagging is done using 12 tags:
        # VERB - verbs
        # NOUN - nouns (common and proper)
        # PRON - pronouns
        # ADJ - adjectives
        # ADV - adverbs
        # ADP - adpositions
        # CONJ - conjunctions
        # DET - determiners
        # NUM - cardinal numbers
        # PRT - particles or other function words
        # X - other: foreign words, typos, abbreviations
        print('Sample tagged input:')
        for pair in tagged_text[:10]:
            print(pair)
    return tagged_text


def get_unique_words(tagged_data):
    words = set()
    for pair in tagged_data:
        words.add(pair[0])
    return list(words)


def get_unique_pos(tagged_data):
    words = set()
    for pair in tagged_data:
        words.add(pair[1])
    return list(words)

# Separate the input text into the training set and validation sets
# based on 80:20 ratio.


def split_sets(input_data):
    training_set = []
    validation_set = []
    total_count = len(input_data)
    training_count = int(total_count * 0.8)
    for word in input_data[:training_count]:
        training_set.append(word)
    for pair in input_data[training_count:]:
        validation_set.append(pair[0])

    print('Total input: ', total_count)
    print('Training set count: ', len(training_set))
    print('Validation set count: ', len(validation_set))

    return (training_set, len(training_set), validation_set, len(validation_set))

# Calculate the probability for a word to be a particular part of speech.


def emission_probability(word, part_of_speech, training_set):
    pairs = []
    # Count the number of times the part of speech is in the training set
    for tagged_pair in training_set:
        if tagged_pair[1] == part_of_speech:
            pairs.append(tagged_pair)

    tag_count = len(pairs)
    # Count the number of times the word is tagged with the part of speech
    word_given_tag_list = []
    for tagged_pair in pairs:
        if tagged_pair[0] == word:
            word_given_tag_list.append(tagged_pair[0])
    word_count = len(word_given_tag_list)

    return (word_count, tag_count)

# Calculate the probability that certain parts of speech follow one another.
# i.e. the probability that a verb follows a noun


def transition_probability(pos1, pos2, training_set):
    tags = []
    pos1_count = 0
    for tagged_pair in training_set:
        tags.append(tagged_pair[1])
        # Count the number of times a tag (pos1) is in the training set.
    for tag in tags:
        if tag == pos1:
            pos1_count += 1
    # Count the number of times pos2 follows pos1.
    pos2_given_pos1_count = 0
    for index in range(len(tags) - 1):
        if tags[index] == pos1 and tags[index + 1] == pos2:
            pos2_given_pos1_count += 1

    return (pos2_given_pos1_count, pos1_count)


def create_probability_matrices(tagged_text):
    unique_pos = set()
    unique_words = set()
    emission_matrix = [[]]

    for tagged_pair in tagged_text:
        unique_pos.add(tagged_pair[1])
        unique_words.add(tagged_pair[0])

    emission_matrix = np.zeros(
        (len(unique_words), len(unique_pos)), dtype='float32')
    transition_matrix = np.zeros(
        (len(unique_pos), len(unique_pos)), dtype='float32')

# Create transition matrix
    print('Creating transition probability matrix ...')
    for i, pos1 in enumerate(list(unique_pos)):
        for j, pos2 in enumerate(list(unique_pos)):
            tran_probability_tuple = transition_probability(
                pos1, pos2, tagged_text)
            tran_probability = tran_probability_tuple[0] / \
                tran_probability_tuple[1]
            transition_matrix[i, j] = tran_probability
    #print('transition matrix: ', transition_matrix)

# Create emission matrix
    print('Creating emission probability matrix ...')
    for i, word in enumerate(list(unique_words)):
        for j, pos in enumerate(list(unique_pos)):
            e_probability_tuple = emission_probability(word, pos, tagged_text)
            e_probability = e_probability_tuple[0] / e_probability_tuple[1]
            emission_matrix[i, j] = e_probability
    #print('emission matrix: ', emission_matrix)

    return (unique_pos, transition_matrix, unique_words, emission_matrix)


def hidden_markov_model(input_words, training_set, t_data_frame, e_data_frame):
    predicted_text = []
    unique_tags = set()
    prev_word = training_set[len(training_set) - 1]
    current_word = ()

    for tagged_pair in training_set:
        unique_tags.add(tagged_pair[1])
    unique_tags = list(unique_tags)

    idx = 0
    while idx != len(input_words):
        current_word = (input_words[idx],)
        # Use prev word to determine most likely POS of current word
        # i.e. use transition probability
        #print('Determining part of speech ...')
        #print('prev word: ', prev_word)
        pos_prob = t_data_frame.loc[prev_word[1]].tolist()
        # Get max probability and index where it occurs
        max = 0
        max_index = 0
        for index, prob in enumerate(pos_prob):
            if prob > max:
                max = prob
                max_index = index
        pos = unique_tags[max_index]
        #print('Predicted POS: ', pos)

        # Tag the current predicted word with the predicted POS
        current_word = current_word + (pos,)

        # Get words already tagged with that POS and append random
        # weighted choice of next word (given POS) to predicted text
        #print('Choosing word based on part of speech ...')
        #print('current word: ', current_word)
        word_prob = e_data_frame[pos].tolist()
        words = e_data_frame.index.tolist()
        #print('Sum of probability: ', sum(word_prob))
        prediction = random.choices(words, weights=word_prob)
        predicted_text.extend(prediction)

        #print('predicted word: ', predicted_text[len(predicted_text) - 1])

        prev_word = current_word
        idx += 1

    return predicted_text


def main():
    tagged_data = tag_parts_of_speech('HarryPotter_Ready.txt')
    # Data Tuple contains (training_set, training_set_size, validation_set, validation_set_size)
    data_tuple = split_sets(tagged_data)
    matrix_data = create_probability_matrices(data_tuple[0])
    matrix_tags = matrix_data[0]
    transition_matrix = matrix_data[1]
    emission_matrix = matrix_data[3]
    # For creating readable table out of probability matrices
    trans_matrix_frame = panda.DataFrame(
        transition_matrix, columns=list(matrix_tags), index=list(matrix_tags))
    e_matrix_frame = panda.DataFrame(emission_matrix, columns=list(
        matrix_tags), index=list(matrix_data[2]))
    # trans_matrix_frame.to_excel('TransitionMatrix.xlsx')
    # e_matrix_frame.to_excel('EmissionMatrix.xlsx')
    #print("Training:", data_tuple[0])
    #print("Testing: " , data_tuple[2])
    #print('transition frame: ', trans_matrix_frame)
    #print('emission frame: ', e_matrix_frame)
    # Test Hidden Markov Model on validation set
    next_word_prediction = hidden_markov_model(
        data_tuple[2], data_tuple[0], trans_matrix_frame, e_matrix_frame)
    #print('Predicted words: ', next_word_prediction)

    # Calculate the Score of the HMM
    score = 0
    for i in range(0, len(next_word_prediction)-1):
        if next_word_prediction[i] == data_tuple[2][i]:
            score = score + 1
        else:
            score = score
    score = score/len(next_word_prediction) * 100
    print('The final score is: ' + str(score) + '%')


if __name__ == "__main__":
    main()
