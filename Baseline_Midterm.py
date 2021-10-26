from random import randint
from termcolor import colored


def pre_processing(filename, output_filename):
    # Open the file specified by filename in utf8 format
    file = open(filename, "r", encoding='utf8')

    # Traversing file line by line
    for line in file:
        # Strip every line of special characters and new line symbol
        stripped_line = line.replace(',', '').replace('"', '').replace("'",'').replace(' .', '.').replace('...', '.').rstrip()
        # Save the output to another file so that we can read it later
        with open(output_filename, 'a') as f:
            f.write(' ' + stripped_line)
            f.close()
    file.close()


def find_words():
    # A file named "processed_training" the training file, will be opened with the
    # reading mode.
    file = open("processed_training.txt", "r", encoding='cp1252')
    all_words = []
    results = []

    # Traversing file line by line
    for line in file:
        # splits each line into words and removing spaces and punctuations from the input
        line_word = line.lower().replace(',', '').replace('"', '').replace("'",'').replace('.','').replace('?','').replace('-','').replace(';', '').split(" ")

        # Adding them to list ALL words so that they can later be counted
        for w in line_word:
            if w is not '':
                all_words.append(w)
        # Adding word one time each to a results list which will only hold unique words
        for word in all_words:
            if word not in results:
                results.append(word)

    # Create a list to keep track of how many times the word appears
    count = [0]*(len(results))

    # Finding the max occured word
    for i in range(0, len(results)):

        # Count each word in the file
        for j in range(0, len(all_words)):
            if(results[i] == all_words[j]):
                count[i] = count[i] + 1
    
    return results, count

def process_test_data():
    file = open("processed_test.txt", "r", encoding='cp1252')
    sentences = []
    fragment = []
    word = []
    split_sentence = []

    # Traversing file line by line
    for line in file:
        # splits each line into words and removing spaces and punctuations from the input
        line = line.lower().split(".")

        for w in line:
            #print(len(w)) 
            if len(w) < 10:
                fragment.append(w)
            elif fragment:
                str = ''.join(fragment)
                fragment = []
                fragment.append(str)
                fragment.append(w)
                fragment = ''.join(fragment)
                sentences.append(fragment)
                fragment = []
            else: 
                sentences.append(w)
    for each in sentences:
        split_each = each.split(' ')
        rnd = randint(0, len(split_each)-1)
        while split_each[rnd] == '':
            rnd = rnd+1
        #print(rnd)
        split_sentence.append(each.replace(split_each[rnd], '_', 1))
        #print(split_sentence)
        word.append(split_each[rnd])
        split_each = []

    return split_sentence, word

def check_word(results, count,split_sentence, word):
    temp_count = []
    temp_results = []
    y = 0
    score = 0

    for line in split_sentence:
        # Make temporary lists that can be editted in this loop
        temp_count = list(count)
        temp_results = list(results)

        line_word = line.lower().replace(',', '').replace('"', '').replace("'",'').replace('.','').replace('?','').replace('-','').replace(';', '').split(" ")

        for w in line_word:
            if w is not '_':
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
        final = line.replace('_', found_word)
        
        if found_word == word[y]:
            score = score + 1
            print(final.replace(' ' + found_word + ' ', colored(' ' + found_word + ' ', 'green')))
        else:
            score = score
            print(final.replace(' ' + found_word + ' ', colored(' ' + found_word + ' ', 'red')) + '; ' + colored('Real Word: ' + word[y], 'blue'))
        # Increment y by 1
        y = y + 1
    
    score = (score/y)*100
    return score


if __name__ == "__main__":
    # Run this to preprcoess the input data file
    #pre_processing('training_input.txt', 'processed_training.txt')
    # Run this to preprcoess the test data file
    #pre_processing('testing_input.txt', 'processed_test.txt')

    # Run this to find the most used words in the file
    results, count = find_words()

    # Run this to read the test data and fill in what we think the last word should be
    split_sentence, word = process_test_data()

    # Run this to check if the word was correct
    score = check_word(results, count, split_sentence, word)
    print('Score is '+ str(score) + '%')
