import sys
sys.path.append("..")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from json import loads, dumps
import pickle

class MaterialTokenClassifier(object):
  def __init__(self):
    pass

  def train(self, neg_path='data/common_nouns.json', pos_path='data/compound_names.txt', clf_path='bin/material_classifier.pkl', cv_path='bin/material_cvectorizer.pkl', cross_validate=False):
    neg_examples = loads(open(neg_path, 'rb').read())['nouns']
    pos_examples =  list(set(unicode(open(pos_path, 'rb').read()).split()))
    self.cv = CountVectorizer(analyzer='char', max_features=500, ngram_range=(1, 4))

    X = self.cv.fit_transform(neg_examples + pos_examples)
    y = [0]*len(neg_examples) + [1]*len(pos_examples)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

    if cross_validate:
      print 'LEARNING CURVE (#TRAINING EXAMPLES):'
      self.material_classifier = SVC(class_weight='balanced', C=1.0)
      print learning_curve(self.material_classifier, X_train, y_train)

      print 'CV REGULARIZATION:'
      for regularization_parameter in [0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        self.material_classifier = SVC(class_weight='balanced', C=regularization_parameter)

        cross_scores = cross_val_score(self.material_classifier, X_train, y_train, cv=5)
        print regularization_parameter, sum(cross_scores) / float(len(cross_scores))

    #Set actual parameters
    regularization_parameter = 1.0
    feature_num = 500
    self.cv = CountVectorizer(analyzer='char', max_features=feature_num, ngram_range=(1, 4))
    self.material_classifier = LogisticRegression(class_weight='balanced', C=regularization_parameter)

    self.cv.fit(neg_examples + pos_examples)
    self.material_classifier.fit(X_train, y_train)
    print self.material_classifier.score(X_test, y_test)
    print precision_recall_fscore_support(y_test, self.material_classifier.predict(X_test), average='binary')

  def predict(self):
    pass

  def predict_one(self, string):
    vector = self.cv.transform([string])
    pred = self.material_classifier.predict(vector)
    return pred[0]

  def save(self):
    pickle.dump(self.material_classifier, open(clf_path, 'wb'))
    pickle.dump(self.cv, open(cv_path, 'wb'))

  def load(self, clf_path='bin/material_classifier.pkl', cv_path='bin/material_cvectorizer.pkl'):
    try:
      self.material_classifier = pickle.load(open(clf_path, 'rb'))
      self.cv = pickle.load(open(cv_path, 'rb'))
      return True
    except:
      return False
