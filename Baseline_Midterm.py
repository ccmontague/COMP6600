from termcolor import colored


def find_words(Input_Filename):
    # Open the input file with reading mode
    file = open(Input_Filename, "r", encoding='utf8')
    all_words = []
    results = []

    # Traversing file line by line
    for line in file:
        # Adding them to list ALL words so that they can later be counted
        for w in line.split():
            all_words.append(w)
            # Adding word one time each to a results list which will only hold unique words
            if w not in results:
                results.append(w)

    # Create a list to keep track of how many times the word appears
    count = [0]*(len(results))

    # Finding the max occured word
    for i in range(0, len(results)):

        # Count each word in the file
        for j in range(0, len(all_words)):
            if(results[i] == all_words[j]):
                count[i] = count[i] + 1

    return results, count


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


if __name__ == "__main__":
    # Input Filename you want to use for the data source
    Input_Filename = "HarryPotter_Ready.txt"
    # Set this to the number of words you would like to use as the input
    Num_Words = 5

    # Run this to find the most used words in the file
    # Takes in:
    #   Input_Filename: the input filename from preprocessing
    # Returns:
    #   results: the list of every word used
    #   count: the list of how many times the word is used
    # The Results and Count list are related such that you can use the same index for both
    results, count = find_words(Input_Filename)

    # Run this to split the input data into sentences with N words and the correct next word
    # Takes in:
    #   Num_Words + 1: the number of words you want to use (the +1 is to account for 0 indexing)
    #   Input_Filename: the input filename from preprocessing
    # Returns:
    #   split_sentence: the list of input sentences for testing the algorithm
    #   word: the list of correct words for scoring the algorithm
    # The split_sentence and word list are related such that you can use the same index for both
    split_sentence, word = process_test_data(Num_Words+1, Input_Filename)

    # Run this to run the baseline algorithm. This algorithm will read the input sentence of N words (split sentence) and remove from the words list any word in that input (results). It will then find the word that remains in the results list with the highest count and input that at the end of the sentence. It will then check that word against the real word (word) and score the algorithm accordingly.
    # Takes in:
    #   results: the list of every word used
    #   count: the list of how many times the word is used
    #   split_sentence: the list of input sentences for testing the algorithm
    #   word: the list of correct words for scoring the algorithm
    # Returns:
    #   score: the percentage that the algorithm inputed the correct word
    score = baseline_algorithm(results, count, split_sentence, word)
    print('Score is ' + str(score) + '%')
