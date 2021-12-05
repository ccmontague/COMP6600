COMP6600 - AI Next Word Prediction Project

Group 6
Courtney Montague - 904212670
Sawyer Hannel - 904005709
Jasmine McIntosh - 902503488
Abriana Fornis - 903469966

The following steps detail how to run the project files in Python.

Baseline Algorithm
1. Run preprocessing.py to format the data such that it can be used by the models.
2. Ensure that a Harry_Potter_Ready.txt file was created in the folder
3. Open the Baseline Algorithm and scroll down to the 
4. Set N equal to the number of words you would like to use as input
5. Run the Baseline_Final.py

LSTM Model
1. Run preprocessing.py to format the data such that it can be used by the models.
2. Ensure that a Harry_Potter_Ready.txt file was created in the folder
3. Run command 'python install -r requirements.txt'
4. Open NextWodPrediction.py and update the input size, epochs, and batch size as desired.
5. Run command 'python NextWordPrediction.py' to generate the tokenizer and model.
6. Open Predictions.py and update the Num_Words for number of words and the paths to the tokenizer and model.
7. Run command 'python Predictions.py'.

Hidden Markov Model (HMM) Algorithm
1. Run preprocessing.py to format the data such that it can be used by the models.
2. Ensure that a Harry_Potter_Ready.txt file was created in the folder
3. Run the HiddenMarkovModel_Final.py

N-Gram Markov Model
1. Run preprocessing.py to format the data such that it can be used by the models.
2. Ensure that a Harry_Potter_Ready.txt file was created in the folder
3. Run N_Gram_Markov.py
