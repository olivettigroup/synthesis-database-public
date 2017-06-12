from base_models import (Document, BasePaper, BaseParagraph, BaseMaterial, BaseOperation, BaseDescriptor, BaseProperty,
                            BaseAmount, BaseCondition, BaseApparatus, BaseConnection)

### PAPER AND PARAGRAPHS ###
class Paper(BasePaper):
  structure = dict(BasePaper.structure, **{
    'operations': ([BaseOperation], []),
    'materials': ([BaseMaterial], []),
    'paragraphs': ([BaseParagraph], []),
    'connections': ([BaseConnection], [])
    })

class Paragraph(BaseParagraph):
  structure = dict(BaseParagraph.structure, **{
  })

### ALGORITHMICALLY EXTRACTED OBJECTS ###
class Operation(BaseOperation):
  structure = dict(BaseOperation.structure, **{
    'type': (unicode, None), #operation type
    'traversals': (int, None), #number of times to apply operation
    'stage': (int, None), #preprocessing, key reaction, postprocessing
  })

class Material(BaseMaterial):
  structure = dict(BaseMaterial.structure, **{
    'alias': (unicode, []), #unique name / formula
    'mpids': ([unicode], []),
    'feature_vector': ([float], []),
    'physical_data': (dict, {}),
  })

class Descriptor(BaseDescriptor):
  structure = dict(BaseDescriptor.structure, **{
    'structure': ([unicode], []) #physical structure, e.g. nanowire
  })

class Property(BaseProperty):
  structure = dict(BaseProperty.structure, **{
    'num_value': (float, None),
    'units': (unicode, None), #e.g. nm
    'type': (unicode, None)
  })

class Amount(BaseAmount):
  structure = dict(BaseAmount.structure, **{
    'num_value': (float, None),
    'units': (unicode, None), #e.g. g, mol, mL
    'misc': (unicode, None),
  })

class Condition(BaseCondition):
  structure = dict(BaseCondition.structure, **{
    'num_value': (float, None),
    'misc': (unicode, None),
    'units': (unicode, None), #e.g. C, RPM
  })

class Apparatus(BaseApparatus):
  structure = dict(BaseApparatus.structure, **{
    'type': (unicode, None), #e.g. autoclave, mill, oven
  })

class Connection(BaseConnection):
  structure = BaseConnection.structure
