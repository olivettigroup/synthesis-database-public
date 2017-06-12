from bson.objectid import ObjectId
from copy import deepcopy
class Document(dict):

  structure = dict()

  def __init__(self, initializer=None):
    for key, (struct, value) in self.structure.iteritems():
      dict.__setitem__(self, key, deepcopy(value))

    if initializer is not None:
      for key, value in initializer.iteritems():
        self.__setitem__(self, key, deepcopy(value))

  def __setitem__(self, key, value):
    if key not in self.structure:
      raise KeyError("Invalid key used: '" + key + "'.")

    expected = self.structure[key][0]

    if type(expected) == list:
      if type(value) != list:
        raise TypeError("Invalid type used: Expected '[" + self.structure[key][0][0].__name__ + "]' but key '" + key + "' is not array.")
      if not all(isinstance(x, expected[0]) for x in value):
        raise TypeError("Invalid type used: Expected '[" + self.structure[key][0][0].__name__ + "]' but got '" + type(value).__name__ + "' for item in key '" + key + "'.")
    elif not isinstance(value, expected):
      raise TypeError("Invalid type used: Expected '" + self.structure[key][0].__name__ + "' but got '" + type(value).__name__ + "' for key '" + key + "'.")

    return dict.__setitem__(self, key, value)

class BasePaper(Document):
  structure = dict(Document.structure, **{
    '_id': (ObjectId, None),
    'doi': (unicode, None),
    'pdf_loc': (unicode, None),
    'html_loc': (unicode, None),
    'watr_json_loc': (unicode, None),
    'title': (unicode, None),
    'abstract': (unicode, None),
    'modified': (int, None)
    })

class BaseParagraph(Document):
  structure = dict(Document.structure, **{
    '_id': (unicode, None),
    'text': (unicode, None), #Post-processed clean text
    'type': (unicode, None), #{intro, recipe, characterization, results, other}
    'watr_text_lines': ([list], None)
    })

class BaseCondition(Document):
  structure = Document.structure

class BaseApparatus(Document):
  structure = Document.structure

class BaseAmount(Document):
  structure = Document.structure

class BaseDescriptor(Document):
  structure = Document.structure

class BaseProperty(Document):
  structure = Document.structure

class BaseOperation(Document):
  structure = dict(Document.structure, **{
    '_id': (unicode, None),
    'conditions': ([BaseCondition], []),
    'apparatuses': ([BaseApparatus], []),
    'order': (int, None)
  })

class BaseConnection(Document):
  structure = dict(Document.structure, **{
    'id1' : (unicode, None),
    'id2' : (unicode, None)
    })

class BaseMaterial(Document):
  structure = dict(Document.structure, **{
    '_id': (unicode, None),
    'amounts' : ([BaseAmount], []),
    'descriptors': ([BaseDescriptor], []),
    'properties': ([BaseProperty], []),
    'is_target': (bool, None)
    })
