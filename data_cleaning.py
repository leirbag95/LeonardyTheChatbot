#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 09:48:17 2019

@author: gabrielelfassi
"""

# return the wordnet object value corresponding to the POS tag

import nltk
import pandas as pd
from nltk.corpus import wordnet
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')
stop = stopwords.words('english')
#!pip install gensim
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
import re


class DataCleaning:

	def __init__(self, data):
		self.data = data


	def data_cleaning(self):
		datac = self.data.copy()
		datac["pattern"].fillna(" ", inplace=True)
		datac["pattern"] = datac["pattern"].apply(lambda x: re.sub(r"[^a-zA-Z0-9]+", ' ', x))
		datac["pattern_cleaned"] = datac["pattern"].apply(lambda x: self.clean_text(x))
#        datac["nb_chars"] = datac["pattern"].apply(lambda x: len(x))
		# datac["nb_word"] = datac["pattern"].apply(lambda x: len(x.split(" ")))
		documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(datac["pattern_cleaned"].apply(lambda x: x.split(" ")))]
		# train a Doc2Vec model with our text data
		model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
		# transform each document into a vector data
		doc2vec_df = datac["pattern_cleaned"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
		doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
		datac = pd.concat([datac, doc2vec_df], axis=1)
		tfidf = TfidfVectorizer(min_df = 1)
		try:
			tfidf_result = tfidf.fit_transform(datac["pattern_cleaned"]).toarray()
			tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
			tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
			tfidf_df.index = datac.index
			datac = pd.concat([datac, tfidf_df], axis=1)
		except:
			print("An error was occured with tfidf")

		return datac

	def clean_text(self, text):
		# lower text
		text = text.lower()

		# tokenize text and remove puncutation
		text = [word.strip(string.punctuation) for word in text.split(" ")]
		# remove words that contain numbers
		text = [word for word in text if not any(c.isdigit() for c in word)]
		# remove stop words
		# text = [x for x in text if x not in stop and stop != "how"]
		# remove empty tokens
		text = [t for t in text if len(t) > 0]
		# pos tag text
		pos_tags = pos_tag(text)
		# lemmatize text
		text = [WordNetLemmatizer().lemmatize(t[0], self.get_wordnet_pos(t[1])) for t in pos_tags]
		# remove words with only one letter
		text = [t for t in text if len(t) > 1]
		# join all
		text = " ".join(text)
		return text

	def get_wordnet_pos(self,pos_tag):
		if pos_tag.startswith('J'):
			return wordnet.ADJ
		elif pos_tag.startswith('V'):
			return wordnet.VERB
		elif pos_tag.startswith('N'):
			return wordnet.NOUN
		elif pos_tag.startswith('R'):
			return wordnet.ADV
		else:
			return wordnet.NOUN
