import string

# Take in the filename input
val = input("Enter the filename you would like to process: ")

# Open the File
file = open(val, "r", encoding="utf8")

# Read the Lines and place them line by line into the line variable
lines = []
for i in file:
    lines.append(i)

# Create an empty variable data and fill it with every line joined together
data = ""
for i in lines:
    data = ' '. join(lines)

# Remove any line breaks, new lines, extra lines, etc.
data = data.replace('\n', '').replace(
    '\r', '').replace('\ufeff', '')

# Find and remove all special characters
translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
data = data.translate(translator)

# Create an empty list z and fill it with one instance of each word
z = []
for i in data.split():
    if i not in z:
        z.append(i)

# Join all of the words in the list z together
new_data = ' '.join(z)

# Open a new file and save the data to it
f = open("processed_data.txt", "w+", encoding="utf8")
f.write(new_data)
f.close()
