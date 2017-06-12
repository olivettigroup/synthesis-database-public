#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("..")

from classifiers.material_token_classifier import MaterialTokenClassifier
from pymatgen import (Composition, Element)
from re import (match, search)

from keras.models import Sequential
from keras.metrics import fmeasure
from keras.layers import Dense, Activation, Merge, Dropout
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2, activity_l2
from keras import backend as K

from gensim.models import Word2Vec
from collections import OrderedDict
from json import loads, dumps
from pymongo import MongoClient
import numpy as np
import tensorflow as tf


class TokenClassifier(object):
  def __init__(self, db):
    token_classes =  {
      0: 'null',
      1: 'amt_unit',
      2: 'amt_misc',
      3: 'cnd_unit',
      4: 'cnd_misc',
      5: 'material',
      6: 'target',
      7: 'operation',
      8: 'descriptor',
      9: 'prop_unit',
      10: 'prop_type',
      11: 'synth_aprt',
      12: 'char_aprt',
      13: 'brand',
      14: 'intrmed',
      15: 'number',
      16: 'meta',
      17: 'ref',
      18: 'prop_misc'
    }
    self.token_classes = {y:x for x,y in token_classes.iteritems()}

    self.db = db
    self.mc = MaterialTokenClassifier()
    self.mc.load()
    self.connection = MongoClient()

    self._load_lexicon()
    self._load_compound_names()
    self._load_embeddings()

    self.model_type = 'hierarchical'

  def build_nn_model(self, model_type='dense_ff', input_dim=1, inner_dim=64, embedding_dim=100, window_size=3):
    model = None

    if model_type == 'dense_ff':
      model = Sequential()
      model.add(Dense(output_dim=inner_dim, input_dim=input_dim, activation="relu"))
      model.add(Dense(output_dim=len(self.token_classes.keys()), activation="softmax"))

    elif model_type == 'hierarchical':
      heuristic_layer = Sequential()
      heuristic_layer.add(Dense(output_dim=inner_dim, input_dim=input_dim, activation="relu", W_regularizer=l2(1.0), activity_regularizer=activity_l2(1.0)))
      heuristic_layer.add(Dropout(0.5))

      embedding_layer = Sequential()
      embedding_layer.add(Dense(output_dim=inner_dim*(window_size+1), input_dim=embedding_dim*(window_size+1), activation="relu", W_regularizer=l2(1.0), activity_regularizer=activity_l2(1.0)))
      embedding_layer.add(Dropout(0.5))

      merge_layer = Merge([heuristic_layer, embedding_layer], mode='concat')

      model = Sequential()
      model.add(merge_layer)
      model.add(Dense(output_dim=len(self.token_classes.keys()), activation="softmax"))

    model.compile(
      optimizer='adam',
      loss='categorical_crossentropy',
      metrics=['categorical_accuracy', self._non_null_accuracy]
    )

    self.model = model
    self.model_type = model_type

  def featurize_embedding(self, token, embedding_dim=100):
    if token in self.embeddings:
      return self.embeddings[token]
    else:
      return np.array([0]*embedding_dim)

  def featurize(self, tok_label):
    tok_vec_dict = OrderedDict()

    #Generic parse/token/grammar features
    tok_vec_dict['is_verb'] = bool(tok_label['pos'] == 'VERB')
    tok_vec_dict['is_noun'] = bool(tok_label['pos'] == 'NOUN')
    tok_vec_dict['is_noun_chunk'] = bool(tok_label['spacy_noun_chunk'] != '')
    tok_vec_dict['is_entity'] = bool(tok_label['spacy_entity'] != '')

    #Domain-specific features
    tok_vec_dict['like_num'] = tok_label['like_num']
    tok_vec_dict['is_stop'] = tok_label['is_stop']
    tok_vec_dict['is_upper'] = bool(tok_label['raw'].isupper() and tok_label['alpha'] == tok_label['raw'])
    tok_vec_dict['is_cde_abbrev'] = tok_label['in_cde_abbrevs']
    tok_vec_dict['is_cde_cem'] = tok_label['in_cde_cems']
    tok_vec_dict['is_break_word'] = bool(tok_label['lemma'] in self.break_words)
    tok_vec_dict['is_formula_like'] = bool(self._is_compound(tok_label['alphanum']))
    tok_vec_dict['is_operation_verb'] = bool(tok_label['lemma'] in self.operations
                                    and tok_label['pos'] == 'VERB')
    tok_vec_dict['is_hydrothermal'] = bool('hydrothermal' in tok_label['raw_subtree'].lower() or 'autoclave' in tok_label['raw_subtree'].lower())
    tok_vec_dict['is_synthesize_operation'] = bool(tok_label['lemma'] in self.operation_categories['synthesize'])
    tok_vec_dict['is_head_synth'] = bool(tok_label['head_lemma'] in self.operation_categories['synthesize'])
    tok_vec_dict['is_head_op'] = bool(tok_label['head_lemma'] in self.operations)
    tok_vec_dict['is_gerund_like'] = bool(tok_label['pos'] == 'NOUN' and tok_label['raw'][:-3] == 'ing')
    tok_vec_dict['is_gerund_operation'] = bool(tok_vec_dict['is_gerund_like'] and tok_label['raw'][:-3] in self.operations)
    tok_vec_dict['is_nums_alphas'] = bool(match('[0-9.]+[A-Za-z]', tok_label['raw']) is not None)
    tok_vec_dict['is_morphology_word'] = bool(tok_label['lemma'] in self.material_descriptors)
    tok_vec_dict['is_generic_material'] = bool(tok_label['lemma'] in self.generic_materials)
    tok_vec_dict['is_in_compound_names'] = bool(tok_label['raw'] in self.compound_names)
    tok_vec_dict['is_amount_unit'] = bool(tok_label['raw'] in self.numeric_amounts or
                                  tok_label['no_nums'] in self.numeric_amounts or
                                  tok_label['alpha'] in self.numeric_amounts)
    tok_vec_dict['is_condition_unit'] = bool(tok_label['raw'] in self.numeric_conditions or
                                  tok_label['no_nums']  in self.numeric_conditions or
                                  tok_label['alpha']  in self.numeric_conditions)
    tok_vec_dict['is_property_unit'] = bool(tok_label['raw'] in self.property_units or
                                  tok_label['no_nums']  in self.property_units or
                                  tok_label['alpha']  in self.property_units)
    tok_vec_dict['is_apparatus_word'] = bool(tok_label['lemma'] in self.apparatuses)
    tok_vec_dict['is_environmental'] = bool(tok_label['lemma'] in self.environment_conditions)
    tok_vec_dict['is_material_blacklist_pattern'] = bool(search('^(\d)*[a-zA-Z]{0,2}$', tok_label['raw']) is not None)
    tok_vec_dict['is_num_range'] = bool(search(u'^(\d)+[–—\-](\d)+$',  tok_label['raw']) is not None)
    tok_vec_dict['is_ph_val'] = bool('pH' in tok_label['raw_ancestors'] and tok_vec_dict['like_num'])
    tok_vec_dict['is_material_like'] = bool(self.mc.predict_one(tok_label['alphanum']) and tok_label['pos'] in ['NOUN', 'PROPN'])
    tok_vec_dict['is_in_pubchem_cache'] = bool(tok_label['pos'] in ['NOUN', 'PROPN'] and self.connection[self.db].chemical_db.find_one({'names':tok_label['raw']}) is not None)

    feature_vec = [v for (k,v) in tok_vec_dict.items()]

    return tok_vec_dict, feature_vec

  def train(self, input_data=[], input_labels=[], batch_size=64, num_epochs=10, checkpt_filepath=None, checkpt_period=5, val_split=0.2, stop_early=True, verbosity=1):
    callbacks = [
    ModelCheckpoint(
      checkpt_filepath,
      monitor='val_loss',
      verbose=0,
      save_best_only=True,
      period=checkpt_period
      )
    ]

    if stop_early:
      callbacks.append(
        EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
      )

    self.model.fit(
      x=input_data,
      y=input_labels,
      batch_size=batch_size,
      nb_epoch=num_epochs,
      validation_split=val_split,
      callbacks= callbacks,
      class_weight={
        0: 0.1,
        1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1,
        10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1,
      },
      verbose=verbosity
    )

  def evaluate(self, inputs, outputs, batch_size=64):
    return self.model.evaluate(inputs, outputs, batch_size=batch_size)

  def predict(self, input_data):
    return [ np.argmax(v) for v in self.fast_predict(input_data + [0])[0] ]

  def save(self, filepath='bin/token_classifier_nn'):
    filepath += '.' + str(self.model_type) + '.model'
    self.model.save(filepath)

  def load(self, filepath='bin/token_classifier_nn'):
    try:
      filepath += '.' + str(self.model_type) + '.model'
      self.model = load_model(filepath, custom_objects={'_non_null_accuracy': self._non_null_accuracy})
      self.fast_predict = K.function(
        self.model.inputs + [K.learning_phase()],
        [self.model.layers[1].output]
      )
    except:
      return -1

  def _load_compound_names(self, fpath='data/compound_names.txt'):
    with open(fpath) as f:
      self.compound_names = unicode(f.read())

  def _load_embeddings(self, fpath='bin/word2vec_embeddings-SNAPSHOT.model'):
    try:
      self.embeddings = Word2Vec.load(fpath)
    except:
      return None

  def _load_lexicon(self, lexicon_path='data/extraction_lexicon.json'):
    with open(lexicon_path) as f:
      self.lexicon = loads(f.read())

    #Known operations to match
    self.operation_categories = self.lexicon['operation_categories']
    self.operations = [] #Build flat list from category list
    self.operations_categories = set() #Build tuple set from category list
    for category in self.operation_categories:
      for operation in self.operation_categories[category]:
        self.operations.append(operation)
        self.operations_categories.add((category, operation))

    #Operating conditions
    self.conditions = []
    self.environment_conditions = self.lexicon['environment_conditions']
    self.numeric_conditions = self.lexicon['numeric_conditions']
    self.conditions.extend(self.environment_conditions)

    #Numeric word conversion
    self.num_words = self.lexicon['num_words']

    #Amount keywords
    self.numeric_amounts = self.lexicon['numeric_amounts']

    #Properties
    self.property_types = self.lexicon['material_properties']
    self.property_units = self.lexicon['property_units']

    #Generic materials
    self.generic_materials = self.lexicon['generic_materials']
    self.material_descriptors = self.lexicon['material_descriptors']

    #Apparatuses
    self.apparatus_categories = self.lexicon['apparatus_categories']
    self.apparatuses = [] #Build flat list from category list
    for category in self.apparatus_categories:
      for apparatus in self.apparatus_categories[category]:
        self.apparatuses.append(apparatus)

    #Break words
    self.break_words = self.lexicon['break_words']

  def _is_compound(self, string):
    try:
      _ = [Element(str(e)) for e in Composition(string).elements]
      if len(_) > 1: return True
    except:
      return False

  def _non_null_accuracy(self, y_true, y_pred):
    return fmeasure(y_true[:,1:], y_pred[:,1:])
