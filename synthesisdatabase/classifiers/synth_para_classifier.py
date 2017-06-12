#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.linear_model import (LogisticRegression)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import (train_test_split)
from sklearn.metrics import (precision_score, recall_score)
from pymongo import MongoClient
from json import (loads)
from autologging import (logged, traced)
import pickle

@logged
class SynthParaClassifier(object):

  def __init__(self, db):
    self.db = db
    self.pred = LogisticRegression(class_weight='balanced')
    self.connection = MongoClient()
    self.threshold = 0.5

  def set_threshold(self, threshold):
    if 0 <= threshold <= 1:
      self.threshold = threshold
      return True
    else:
      return False

  def load_training_data(self, pos_path='data/recipe_paragraphs.txt', neg_path='data/non_recipe_paragraphs.txt'):
    self.training_methods_X = []
    self.training_methods_Y = []

    self.cv = CountVectorizer(
      decode_error='ignore',
      max_features=100,
      stop_words='english',
      binary=False
    )

    all_lines = []
    with open(pos_path, 'rb') as f:
      for line in f: all_lines.append(line)
    with open(neg_path, 'rb') as f:
      for line in f: all_lines.append(line)

    self.cv.fit(all_lines)

    self.__logger.info( 'Loading training data...' )

    with open(pos_path, 'rb') as f:
      for line in f:
        self.training_methods_X.append(self.featurize(line))
        self.training_methods_Y.append(1)

    with open(neg_path, 'rb') as f:
      for line in f:
        self.training_methods_X.append(self.featurize(line))
        self.training_methods_Y.append(0)

  def featurize(self, paragraph):
    numbers = sum(c.isdigit() for c in paragraph)
    letters = sum(c.isalpha() for c in paragraph)

    features = [
      #Character length thresholds
      bool(len(paragraph) > 10),
      bool(len(paragraph) > 100),
      bool(len(paragraph) > 1000),
      bool(len(paragraph) > 10000),

      #Number and word counts
      bool(numbers >= 1),
      bool(numbers > 10),
      bool(numbers > 100),
      bool(letters >= 1),
      bool(letters > 10),
      bool(letters > 100),
      bool(letters > 1000),
      bool(letters > 10000),

      #Special characters
      bool( sum(c in ['(', ')'] for c in paragraph) >= 1),
      bool( sum(c in ['(', ')'] for c in paragraph) >= 10),
      bool( sum(c in ['(', ')'] for c in paragraph) >= 100),
      bool( sum(c in ['[', ']'] for c in paragraph) >= 1),
      bool( sum(c in ['[', ']'] for c in paragraph) >= 10),
      bool( sum(c in ['[', ']'] for c in paragraph) >= 100),

      #Heading heuristics
      bool('experiment' in paragraph[:50].lower()),
      bool('synthesi' in paragraph[:50].lower()),
      bool('prepar' in paragraph[:50].lower()),
      bool('abstract' in paragraph[:50].lower()),
      bool('characteriz' in paragraph[:50].lower()),

      #Domain keyword checks
      bool('heat' in paragraph.lower()),
      bool('dissolve' in paragraph.lower()),
      bool('mix' in paragraph.lower()),
      bool('obtain' in paragraph.lower()),

      bool('mol ' in paragraph.lower()),
      bool('%' in paragraph.lower()),
      bool('ml ' in paragraph.lower()),
      bool(' ph ' in paragraph.lower()),

      bool('ratio' in paragraph.lower()),
      bool('stoichiometric' in paragraph.lower()),

      bool('sample' in paragraph.lower()),
      bool('solution' in paragraph.lower()),
      bool('product' in paragraph.lower()),
      bool('chemical' in paragraph.lower()),

      bool('study' in paragraph.lower()),
      bool('method' in paragraph.lower()),
      bool('technique' in paragraph.lower()),
      bool('route' in paragraph.lower()),

      #Heuristic phrases
      bool('was prepared by' in paragraph.lower()),
      bool('dissolved in' in paragraph.lower()),
      bool('final product' in paragraph.lower()),
      bool('the precursors' in paragraph.lower()),
      bool('purchased from' in paragraph.lower()),

      #Characterization
      bool('xrd' in paragraph.lower()),
      bool('ftir' in paragraph.lower()),
      bool('voltammetry' in paragraph.lower()),
      bool('sem ' in paragraph.lower()),
      bool('microscop' in paragraph.lower()),
      bool('spectroscop' in paragraph.lower()),
    ]

    cv_features = list(self.cv.transform([paragraph]).toarray()[0])

    return features + cv_features

  def train(self, split):
    self.testing_methods_X = []
    self.testing_methods_Y = []

    self.__logger.info( 'Splitting data into training/testing: ' + str(1-split) + '/' + str(split) + '... ' )
    self.training_methods_X, self.testing_methods_X, self.training_methods_Y, self.testing_methods_Y = train_test_split(self.training_methods_X, self.training_methods_Y, test_size=split)

    self.__logger.info( 'Training logistic regression classifier... ' )
    self.pred.fit(self.training_methods_X, self.training_methods_Y)

  def test(self):
    self.__logger.info( 'Testing LR classifier accuracy...' )

    self.accuracy = round(self.pred.score(self.testing_methods_X, self.testing_methods_Y), 2)

    pred_testing_methods_Y = self.pred.predict_proba(self.testing_methods_X)
    pred_testing_methods_Y_class = []

    for item in pred_testing_methods_Y:
      if item[1] > self.threshold:
        pred_testing_methods_Y_class.append(1)
      else:
        pred_testing_methods_Y_class.append(0)

    self.precision = round(precision_score(self.testing_methods_Y, pred_testing_methods_Y_class), 2)
    self.recall = round(recall_score(self.testing_methods_Y, pred_testing_methods_Y_class), 2)

    print 'Accuracy is ' + str(self.accuracy)
    print 'Precision is ' + str(self.precision)
    print 'Recall is ' + str(self.recall)

  def predict_one(self, paragraph_text):
    paragraph_feature_vector = self.featurize(paragraph_text)
    return bool(self.pred.predict_proba([paragraph_feature_vector])[0][1] > self.threshold)

  def save(self, filename):
    try:
      pickle.dump(self.pred, open(filename, 'wb'))
      pickle.dump(self.cv, open(filename + '_cv', 'wb'))
      return True
    except:
      return False

    return False

  def load(self, filename):
    try:
      self.pred = pickle.load(open(filename, 'rb'))
      self.cv = pickle.load(open(filename + '_cv', 'rb'))
      return True
    except:
      return False

    return False
