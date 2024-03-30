import random
import json 
import pickle
import numpy as np
import nltk 
import pandas as pd
import tensorflow  as tf
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

intents = json.loads(open('intents.json').read())


lemmatizer = WordNetLemmatizer()

# Initialize empty lists for words, classes, and documents
words = []
classes = []
documents = []

# Define a list of characters to ignore
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:

        word_list = nltk.word_tokenize(pattern)

        words.extend(word_list)

        documents.append((word_list, intent['tag']))
        

        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]

words = sorted(set(words)) 
print(words)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))



training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([np.array(bag), np.array(output_row)])  # Convert bag and output_row to numpy arrays
    
# random.shuffle(training)
max_length = max(len(bag) for bag, _ in training)

# Pad all 'bag' arrays to the maximum length with zeros
for i in range(len(training)):
    bag, output_row = training[i]
    if len(bag) < max_length:
        padding = np.zeros(max_length - len(bag))
        training[i] = (np.concatenate((bag, padding)), output_row)

# Now, you can convert 'training' to a NumPy array without errors
training = np.array(training)
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print(f"training: {training} \n train_x: {train_x} \n train_y: {train_y}" )

model  = Sequential() 
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu')) 
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print('Model created')
