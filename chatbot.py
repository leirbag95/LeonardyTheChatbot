#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 11:39:08 2019

@author: gabrielelfassi
"""


# libraries
import math
import numpy as np
import pandas as pd
import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('sentiwordnet')
# nltk.download('averaged_perceptron_tagger')
from nltk.tag import UnigramTagger
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import regexp_tokenize
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import re
from nltk.stem import PorterStemmer
#!pip install textblob
from textblob import TextBlob
from textblob import Word
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import data_cleaning as DC
import json
import random


class Leonardy:

	# reset var: allow to train the model and create a new model.h5 file
	def __init__(self, reset=False):
		self.dataframe = pd.read_csv('src/data/intents.csv',sep=';')
		with open('src/data/answers.json') as json_file:
			self.dataframe_answer = json.load(json_file)

		dc = DC.DataCleaning(self.dataframe)
		self.dataframe = dc.data_cleaning()
		self.supervised_model = Supervised(self.dataframe, "tag_index", ["tag_index", "tag", "pattern", "pattern_cleaned"])
		if reset:
			self.supervised_model.create_model()
		from keras.models import load_model
		self.model = load_model('chatbot_model.h5')



	def get_index_score(self, message):
		x_test = pd.DataFrame({"pattern":[message], "tag":["_blank"], "tag_index":1})
		dc=DC.DataCleaning(x_test)
		x_test=dc.data_cleaning()
		features = self.supervised_model.features
		for col in self.dataframe.columns:
			if not col in x_test.columns:
				x_test[col] = 0
			else:
				val = x_test[col]
				x_test = x_test.drop(col, axis=1)
				x_test[col] = val
		pred = self.model.predict(x_test[features])
		# if score pred is lower than 0.60 return nonresponse class
		score = pred[0][self.model.predict_classes(x_test[features])]

		if score < 0.50:
			return ("noanswer",1 - score)
		tag_index = self.model.predict_classes(x_test[features])[0]
		return (self.dataframe[self.dataframe[self.supervised_model.label] == tag_index]["tag"].iloc[0], score)

	def get_response(self, message):
		index, score = self.get_index_score(message)
		print(self.dataframe_answer)
		answers = self.dataframe_answer[index]
		return (random.choice(answers), score)




class Supervised:
	def __init__(self,dataframe, label, ignore_cols):
		self.dataframe = dataframe
		self.label = label
		self.ignore_cols = ignore_cols
		self.features = [c for c in dataframe.columns if c not in ignore_cols]



	def create_model(self):
		from keras.utils import to_categorical
		from sklearn.model_selection import train_test_split
		# split the data into train and test
		X_train, y_train = self.dataframe[self.features], self.dataframe[self.label]
		print(self.dataframe[self.label].value_counts())
		y_train = to_categorical(y_train, num_classes=len(self.dataframe[self.label].value_counts()))
		from keras.models import Sequential
		from keras.layers import Dense
		from keras.layers import Embedding
		from keras.layers import LSTM
		from keras.optimizers import SGD
		from keras.layers import Dropout
		#create model
		model = Sequential()
		#get number of columns in training data
		n_cols_2 = X_train.shape[1]
		#add layers to model

		model.add(Dense(256, input_shape=(n_cols_2,), activation='relu'))
		model.add(Dropout(0.2))
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.2))
		model.add(Dense(len(self.dataframe[self.label].value_counts()), activation='softmax', kernel_initializer='random_normal'))
		#compile model using accuracy to measure model performance
		sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
		#train model
		from keras.callbacks import ModelCheckpoint, EarlyStopping

		model.summary()
		history = model.fit(X_train, y_train, epochs=1000, batch_size=32)
		model.save('chatbot_model.h5', history)


	def mse_metric(self, model, x, y):
		from sklearn.metrics import mean_squared_error
