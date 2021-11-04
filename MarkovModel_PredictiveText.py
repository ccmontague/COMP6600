import numpy as np
import re
import io
import os
from shutil import copyfile

pageNumberRegex = '^Page [0-9]+ \| [a-zA-Z]+'

# Remove lines that include page numbers
def processInputFile(filename):
	path = os.path.dirname(os.path.abspath(__file__))
	sourcePath = os.path.join(path, filename)
	print('sourcePath = ', sourcePath)
	destinationPath = os.path.join(path, 'ModifiedTextInput.txt')
	print('destinationPath = ', destinationPath)

	copyfile(sourcePath, destinationPath)

	with io.open('ModifiedTextInput.txt', encoding='utf8', mode='r') as harryPotter:
		lines = harryPotter.readlines()
	#with io.open('ModifiedTextInput.txt', encoding='utf8', mode='w') as harryPotter:
		for line in lines:
			line.strip()
			if line.__contains__('\n'):
  				line = line.replace('\n', '')
			if re.match(pageNumberRegex, line):
				lines.remove(line)
	#modifiedText = io.open('ModifiedInputText.txt', encoding='utf8', mode='r')
	return lines

def makePairs(corpus):
	for i in range(len(corpus)-1):
		yield (corpus[i], corpus[i+1])

def predictiveText():
	corpus = processInputFile('HarryPotter.txt')
	makePairs(corpus)    
	pairs = makePairs(corpus)

	wordDict = {}

	for word1, word2 in pairs:
			if word1 in wordDict.keys():
					wordDict[word1].append(word2)
			else:
					wordDict[word1] = [word2]
	
	firstWord = np.random.choice(corpus)

	while firstWord.islower():
			firstWord = np.random.choice(corpus)

	chain = [firstWord]

	numWordsToSimulate = 50

	for i in range(numWordsToSimulate):
			chain.append(np.random.choice(wordDict[chain[-1]]))

	' '.join(chain)
	print(chain)

predictiveText()