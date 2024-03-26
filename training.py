import random
import json 
import pickle
import numpy as np
import nltk 
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# Load the intents from the JSON file
intents = json.loads(open('intents.json').read())

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize empty lists for words, classes, and documents
words = []
classes = []
documents = []

# Define a list of characters to ignore
ignore_letters = ['?', '!', '.', ',']

# Iterate through each intent in the intents JSON
for intent in intents['intents']:
    # Iterate through each pattern in the intent
    for pattern in intent['patterns']:
        # Tokenize the pattern into a list of words
        word_list = nltk.word_tokenize(pattern)
        # Extend the words list with the word_list
        words.extend(word_list)
        # Append the word_list and intent tag as a tuple to the documents list
        documents.append((word_list, intent['tag']))
        
        # If the intent tag is not already in the classes list, add it
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize the words and remove the ignore_letters
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
# Sort and remove duplicates from the words list
words = sorted(set(words)) 

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Print the words list
print(words)
