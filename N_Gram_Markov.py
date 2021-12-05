from random import randint
import numpy as np

# Separate the input text into the training set and validation sets
# based on 80:20 ratio.
def split_sets(filename):
    with open(filename, 'r') as input_text:
        input_string = input_text.read()
        input_data = input_string.split()
    
    training_set = []
    validation_set = []
    total_count = len(input_data)
    training_count = int(total_count * 0.8)
    for word in input_data[:training_count]: 
        training_set.append(word)
    for pair in input_data[training_count:]:
  	    validation_set.append(pair)
  
    #print('Total input: ', total_count)
    #print('Training set count: ', len(training_set))
    #print('Validation set count: ', len(validation_set))

    return training_set, validation_set,

def create_N_gram_dict(training_set, N):
    N_gram_dictionary = {}
    Single_gram_dictionary = {}

    # Create N Gram Dict
    for k in range(len(training_set) - N):
        n_gram = []
        for w in range(0, N):
            n_gram.append(training_set[k + w])
        n_gram = ' '.join(n_gram)
        update_dict(n_gram, training_set[k+N], N_gram_dictionary)

    # Create Single Gram Dict
    for i in range(len(training_set) - 1):
        update_dict(training_set[i], training_set[i + 1], Single_gram_dictionary)

    return N_gram_dictionary, Single_gram_dictionary

def update_dict(current : str, next_word : str, dictionary):

    if current not in dictionary:
        dictionary.update({current: {next_word: 1} })
        return
    
    options = dictionary[current]

    if next_word not in options:
        options.update({next_word : 1})
    else:
        options.update({next_word : options[next_word] + 1})

    dictionary[current] = options

    return dictionary

def adjust_prob(dictionary):
    
    for word, transition in dictionary.items():
        transition = dict((key,value / sum(transition.values())) for key, value in transition.items())
        dictionary[word] = transition

    return dictionary

def Markov_Model(N_Gram_Dict, Single_Gram_Dict, test_input, N):
    predicted = []
    line = []
    i = 0
    # Split the Test Input into N Word Lines
    for w in range(0, len(test_input)):
        lines = ' '.join(test_input[w])
        line.append(lines)
    # Search for Lines in the N Gram Dictionary and Single Gram Dictionary
    for l in line:
        # Search in N Gram Dictionery - If its not there search the Single Gram Dictionary
        if l not in N_Gram_Dict:
            l = l.strip().split(' ')[-1]
            if l not in Single_Gram_Dict:
                predicted.append(' ')
            else:
                options = Single_Gram_Dict[l]
                predicted.append(np.random.choice(list(options.keys()), p=list(options.values())))
        else:  
            options = N_Gram_Dict[l]
            predicted.append(np.random.choice(list(options.keys()), p=list(options.values())))
        i += 1
    
    return predicted

def get_test_set(test_input, N):
    words = []
    correct_words = []
    for i in range(0, int(len(test_input)*.2)):
        rnd = randint(0, len(test_input)-(N+1))
        words.append(test_input[rnd:rnd+N])
        # Get the correct words by taking the Nth word from the list
        correct_words.append(test_input[rnd+N])

    return words, correct_words

def score_function(prediction, correct_words):
    # Calculate the Score of the HMM
    score = 0
    for i in range(0, len(prediction)):

        if prediction[i] == correct_words[i]:
            score = score + 1
        else:
            score = score
    score = score/len(prediction) * 100
    print('The final score is: ' + str(score) + '%')


def main():
    filename = 'HarryPotter_Ready.txt'
    # Separate the input text into the training set and validation sets
    # based on 80:20 ratio.
    training_set, validation_set = split_sets(filename)
    # N Gram Modeling - Choose the Number of Words in the Input
    N = 5
    # Create the Dictionary of Words - There will be an N Gram Dictionary and a Single Gram Dictionary
    N_gram, Single_gram = create_N_gram_dict(training_set, N)
    # Adjust the Porbabilites to a 0 to 1 scale
    N_gram = adjust_prob(N_gram)
    Single_gram = adjust_prob(Single_gram)
    #print(N_gram)
    #print(Single_gram)

    # Seperate the Validation Set into N word inputs and their correct next word
    words, correct_words = get_test_set(validation_set, N)
    
    # Take in the N word input from the validation set and predict the next word
    prediction = Markov_Model(N_gram, Single_gram, words, N)
    print('The predicted words are: ', prediction)
    print('The correct words are: ', correct_words)

    # Score how well the model does
    score_function(prediction, correct_words)


if __name__ == "__main__":
  main()
