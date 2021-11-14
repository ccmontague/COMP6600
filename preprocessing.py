### This script will be used for the preprocessing of the data ####
###################################################################
import os
import string


def remove_chapter_names(filename, output_filename):
    # Open the file specified by filename in utf8 format
    file = open(filename, "r", encoding='utf8')

    # Check if a cleaned Harry Potter File already exists
    # If it exists, remove it, if it does not, move on.
    if os.path.exists(output_filename):
        print("Removing File: " + str(output_filename))
        os.remove(output_filename)
    else:
        print("The file " + str(output_filename) + " does not exist")

    # Traversing input file line by line
    special_characters = '"!@#$%^&*()-+?_=,<>/"'
    for line in file:
        if line.isupper() and not any(c in special_characters for c in line):
            print('Removing: ' + line)
        elif "Page" in line:
            print('Removing: ' + line)
        else:
            with open(output_filename, 'a', encoding='utf8') as f:
                f.write(line)
                f.close()
    file.close()


def remove_special_characters(filename, output_filename):
    # Open the file specified by filename in utf8 format
    file = open(filename, "r", encoding='utf8')

    # Check if a cleaned Harry Potter File already exists
    # If it exists, remove it, if it does not, move on.
    if os.path.exists(output_filename):
        print("Removing File: " + str(output_filename))
        os.remove(output_filename)
    else:
        print("The file " + str(output_filename) + " does not exist")

    lines = []

    # Traversing file line by line
    for line in file:
        lines.append(line)

    # Create data variable that is one large string
    data = ""
    for i in lines:
        data = ' '.join(lines)

    # Replace all special characters
    data = data.lower().replace('\n', '').replace(
        '\r', '').replace('\ufeff', '').replace('”', '').replace("’", "").replace('“', '').replace("‘", "").replace('—', '')

    # Remove punctuation
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    data = data.translate(translator)

    # Remove unwanted spaces
    all_words = []
    for i in data.split():
        # Keep only one occurence of the word
        # if i not in z:
        all_words.append(i)

    # Join all of the words together again to write to the new text file
    data = ' '.join(all_words)

    # Write to the new text file
    with open(output_filename, 'a', encoding='utf8') as f:
        f.write(data)
        f.close()

def remove_all_but_periods(filename, output_filename):
    # Open the file specified by filename in utf8 format
    file = open(filename, "r", encoding='utf8')

    # Check if a cleaned Harry Potter File already exists
    # If it exists, remove it, if it does not, move on.
    if os.path.exists(output_filename):
        print("Removing File: " + str(output_filename))
        os.remove(output_filename)
    else:
        print("The file " + str(output_filename) + " does not exist")

    lines = []

    # Traversing file line by line
    for line in file:
        lines.append(line)

    # Create data variable that is one large string
    data = ""
    for i in lines:
        data = ' '.join(lines)

    # Replace all special characters
    data = data.lower().replace('\n', '').replace(
        '\r', '').replace('\ufeff', '').replace('”', '').replace("’", "").replace('“', '').replace("‘", "").replace('—', '').replace(',','').replace(';','').replace('/','')

    # Replace punctations with periods
    translator = str.maketrans(string.punctuation, '.'*len(string.punctuation))
    data = data.translate(translator)

    # Remove unwanted spaces
    all_words = []
    for i in data.split():
        # Keep only one occurence of the word
        # if i not in z:
        all_words.append(i)

    # Join all of the words together again to write to the new text file
    data = ' '.join(all_words)

    # Write to the new text file
    with open(output_filename, 'a', encoding='utf8') as f:
        f.write(data)
        f.close()


if __name__ == "__main__":
    # Run this to clean the HarryPotter.txt file of Chapter Names and Page Numbers
    remove_chapter_names('HarryPotter.txt', 'HarryPotter_Clean.txt')
    # Run this to remove any special characters and make all characters lowercase
    remove_special_characters('HarryPotter_Clean.txt', 'HarryPotter_Ready.txt')
    # Run this to remove all special characters and replace with periods. this will allow the HMM to know when a sentence ends for emission probability purposes.
    remove_all_but_periods('HarryPotter_Clean.txt', 'HarryPotter_Ready_Periods.txt')
