from random import randint
from termcolor import colored


def find_words():
    # A file named "processed_training" the training file, will be opened with the
    # reading mode.
    file = open("HarryPotter_Ready.txt", "r", encoding='utf8')
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


def process_test_data(N):
    file = open("HarryPotter_Ready.txt", "r", encoding='utf8')
    word = []
    split_sentence = []
    sentence = []
    i = 0
    n = 0

    # Traversing file line by line
    for line in file:
        # splits each line into words and removing spaces and punctuations from the input
        for w in line.split():
            i = i + 1
            sentence.append(w)
            if i == (N*n) + N:
                n = n + 1
                i = N*n
                if N*n == 0:
                    sentence.insert(N-1, '_')
                    test_input = ' '.join(sentence[0:N])
                    word.append(sentence[N])
                else:
                    sentence.insert((N*n)-1, '_')
                    test_input = ' '.join(sentence[N*(n-1):N*n])
                    word.append(sentence[N*n])
                split_sentence.append(test_input)
                # print(split_sentence)

    return split_sentence, word


def check_word(results, count, split_sentence, word):
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
        #final = line.replace('_', found_word)

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
    # Run this to find the most used words in the file
    results, count = find_words()
    # print(results)
    # print(count)

    # Run this to read the test data and fill in what we think the last word should be
    Num_Words = 10
    split_sentence, word = process_test_data(Num_Words)

    # Run this to check if the word was correct
    score = check_word(results, count, split_sentence, word)
    print('Score is ' + str(score) + '%')
