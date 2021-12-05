from termcolor import colored
import matplotlib.pyplot as plt

# Separate the input text into the training set and validation sets
# based on 80:20 ratio.


def split_sets(Input_Filename):
    with open(Input_Filename, 'r') as input_text:
        input_string = input_text.read()
        words = input_string.split()
        training_set = []
        validation_set = []
        total_count = len(words)
        training_count = int(total_count * 0.8)
        for word in words[:training_count]:
            training_set.append(word)
        for pair in words[training_count:]:
            validation_set.append(pair)

    return (training_set, validation_set)


def find_words(training_set):
    results = []

    for w in training_set:
        # Adding word one time each to a results list which will only hold unique words
        if w not in results:
            results.append(w)

    # Create a list to keep track of how many times the word appears
    count = [0]*(len(results))

    # Finding the max occured word
    for i in range(0, len(results)):

        # Count each word in the file
        for j in range(0, len(training_set)):
            if(results[i] == training_set[j]):
                count[i] = count[i] + 1

    return results, count


def process_test_data(N, testing_set):
    word = []
    sentence = []
    split_sentence = []
    i = 0
    # Traversing file line by line
    for f in range(0, len(testing_set)):
        sentence.append(testing_set[f])
        i = i + 1
        if i == N:
            i = 0
            sentence.insert(N-1, '_')
            test_input = ' '.join(sentence[0:N])
            word.append(sentence[N])
            split_sentence.append(test_input)
            sentence = []

    return split_sentence, word


def baseline_algorithm(results, count, split_sentence, word):
    temp_count = []
    temp_results = []
    y = 0
    score = 0

    for line in split_sentence:
        # Make temporary lists that can be editted in this loop
        temp_count = list(count)
        temp_results = list(results)

        for w in line.split():
            while w in temp_results:
                idx = temp_results.index(w)
                temp_results.remove(w)
                temp_count.pop(idx)

        # Find the most common word and its index number
        max_value = max(temp_count)
        max_index = temp_count.index(max_value)
        # Use the index number to find the most frequent word in the list
        found_word = temp_results[max_index]
        # Remove this word from the count and results in case we need to loop again
        temp_results.remove(found_word)
        temp_count.pop(max_index)
        # Checks if the found word was the correct word and prints the output
        if found_word == word[y]:
            score = score + 1
            print(line.replace('_', colored(found_word, 'green')))
        else:
            score = score
            print(line.replace('_', colored(found_word, 'red')) +
                  '; ' + colored('Real Word: ' + word[y], 'blue'))
        # Increment y by 1
        y = y + 1

    score = (score/y)*100
    return score


def plot_most_common(results, count):
    top_ten_words = []
    top_ten_count = []
    # Make temporary lists that can be editted in this loop
    temp_count = list(count)
    temp_results = list(results)
    for i in range(0, 10):
        # Find the most common word and its index number
        max_value = max(temp_count)
        top_ten_count.append(max_value)
        max_index = temp_count.index(max_value)
        # Use the index number to find the most frequent word in the list
        found_word = temp_results[max_index]
        top_ten_words.append(found_word)
        # Remove this word from the count and results in case we need to loop again
        temp_results.remove(found_word)
        temp_count.pop(max_index)
    # Plot the results
    x_pos = [i for i, _ in enumerate(top_ten_words)]

    plt.bar(x_pos, top_ten_count, color='green')
    plt.xlabel("Words")
    plt.ylabel("Count")
    plt.title("Top 10 Most Frequent Words Found")

    plt.xticks(x_pos, top_ten_words)

    plt.show()


if __name__ == "__main__":
    # Input Filename you want to use for the data source
    Input_Filename = "HarryPotter_Ready.txt"
    # Set this to the number of words you would like to use as the input
    # Ex: [1, 2, 3, 4, 5] will run the algorithm using 1 - 5 words as the input and then plot the results
    Num_Words = [5]
    # Creates an empty score array
    score_array = []

    for n in Num_Words:
        # Run this to split the input file into the training set and the testing set. We decided to use an 80:20 split meaning that 80% of the input data will be used for training and 20% will be used for testing.
        # Takes In:
        #   Input_Filename: the input filename from preprocessing
        # Returns:
        #   training_set: 80% of the input data split into words used for training
        #   testing_set: 20% of the input data split into words used for testing
        training_set, testing_set = split_sets(Input_Filename)

        # Run this to find the most commonly used words in the file
        # Takes in:
        #   training_set: 80% of the input data split into words
        # Returns:
        #   results: the list of every word used
        #   count: the list of how many times the word is used
        # The Results and Count list are related such that you can use the same index for both
        results, count = find_words(training_set)

        # Run this function to create a bar chart of the most common words found
        plot_most_common(results, count)

        # Run this to split the input data into sentences with N words and the correct next word
        # Takes in:
        #   Num_Words + 1: the number of words you want to use (the +1 is to account for 0 indexing)
        #   testing_set: 20% of the input data split into words
        # Returns:
        #   split_sentence: the list of input sentences for testing the algorithm
        #   word: the list of correct words for scoring the algorithm
        # The split_sentence and word list are related such that you can use the same index for both
        split_sentence, word = process_test_data(n+1, testing_set)

        # Run this to run the baseline algorithm. This algorithm will read the input sentence of N words (split sentence) and remove from the words list any word in that input (results). It will then find the word that remains in the results list with the highest count and input that at the end of the sentence. It will then check that word against the real word (word) and score the algorithm accordingly.
        # Takes in:
        #   results: the list of every word used
        #   count: the list of how many times the word is used
        #   split_sentence: the list of input sentences for testing the algorithm
        #   word: the list of correct words for scoring the algorithm
        # Returns:
        #   score: the percentage that the algorithm inputed the correct word
        score = baseline_algorithm(results, count, split_sentence, word)
        print('Score using N = ' + str(n) + ' words is ' + str(score) + '%')
        score_array.append(score)
    if len(Num_Words) > 1:
        plt.plot(Num_Words, score_array)
        plt.title('Score Results by Number of Words in Input')
        plt.xlabel('Number of Words')
        plt.ylabel('Score (Percent)')
        plt.show()
