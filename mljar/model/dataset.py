from marshmallow import Schema, fields, post_load

from base import BaseModel

class DatasetSchema(Schema):
    hid = fields.Str()
    title = fields.Str()
    scope = fields.Str()
    created_at = fields.DateTime(allow_none=True)
    created_by = fields.Number(allow_none=True)
    parent_project = fields.Number(allow_none=True)
    data_type = fields.Str()
    dataset_hash = fields.Str()
    file_name = fields.Str()
    file_path = fields.Str()
    file_size = fields.Str()
    meta = fields.List(fields.Dict(), allow_none=True)
    prediction_only = fields.Number()
    accepted = fields.Number()
    checked = fields.Number()
    derived = fields.Number()
    valid = fields.Number()
    text_msg = fields.Str(allow_none=True)
    column_usage_min = fields.Dict(allow_none=True)

    @post_load
    def make_project_instance(self, data):
        return Dataset(**data)
'''
{u'prediction_only': 0, u'accepted': 1, u'checked': 1, u'data_type': u'tabular', u'title': u'train', u'dataset_hash': u'5127009536139176676',
u'file_name': u'adult_na.csv', u'meta': [], u'derived': 0, u'valid': 1, u'hid': u'NYwvZA3d8Aol', u'file_size': u'3.79', u'scope': u'private', u'text_msg': None,
 u'column_usage_min': {u'cols_to_fill_na': [u'workclass', u'occupation', u'native-country'],
 u'use': [u'age', u'workclass', u'fnlwgt', u'education', u'education-num', u'marital-status', u'occupation', u'relationship', u'race',
  u'sex', u'capital-gain', u'capital-loss', u'hours-per-week', u'native-country'], u'dont': [], u'cols_to_convert_categorical':
   [u'workclass', u'education', u'marital-status', u'occupation', u'relationship', u'race', u'sex', u'native-country', u'income'], u'id': [],
   u'target': [u'income']}, u'file_path': u'users/user-id=1/project-id=AyBz36V93GEX/datasets/dataset-2753725c-248f-401b-b091-db8f5943c11cadult_na.csv'}]
'''
class Dataset(BaseModel):
    schema = DatasetSchema(strict=True)

    def __init__(self, hid, title, scope, data_type,
                    file_name, file_path, file_size, meta, prediction_only,
                    accepted, checked, derived, valid, text_msg, dataset_hash,
                    column_usage_min, created_at = None, created_by = None, parent_project = None):
        self.hid = hid
        self.title = title
        self.scope = scope
        self.created_at = created_at
        self.created_by = created_by
        self.parent_project = parent_project
        self.data_type = data_type
        self.dataset_hash = dataset_hash
        self.file_name = file_name
        self.file_path = file_path
        self.file_size = file_size
        self.meta = meta
        self.prediction_only = prediction_only
        self.accepted = accepted
        self.checked = checked
        self.derived = derived
        self.valid = valid
        self.text_msg = text_msg
        self.column_usage_min = column_usage_min

    def show(self):
        print '-'*50,'\Data details (', self.hid,')\n','-'*50
        print 'Title:', self.title
        print 'File:', self.file_name
        print 'File size:', self.file_size
        print 'Accepted column usage:', self.accepted
        print '-'*50

'''

[{u'prediction_only': 0, u'accepted': 1, u'checked': 1, u'data_type': u'tabular',
u'title': u'train', u'dataset_hash': u'5127009536139176676', u'file_name': u'adult_na.csv',
u'meta': [], u'derived': 0, u'valid': 1, u'hid': u'NYwvZA3d8Aol', u'file_size': u'3.79', u'scope': u'private', u'text_msg': None,
u'column_usage_min':
 {u'cols_to_fill_na': [u'workclass', u'occupation', u'native-country'],
 u'use': [u'age', u'workclass', u'fnlwgt', u'education', u'education-num', u'marital-status', u'occupation', u'relationship', u'race', u'sex',
  u'capital-gain', u'capital-loss', u'hours-per-week', u'native-country'], u'dont': [],
  u'cols_to_convert_categorical': [u'workclass', u'education', u'marital-status', u'occupation', u'relationship', u'race', u'sex', u'native-country', u'income'],
   u'id': [], u'target': [u'income']},
   u'file_path': u'users/user-id=1/project-id=AyBz36V93GEX/datasets/dataset-2753725c-248f-401b-b091-db8f5943c11cadult_na.csv'}]

'''
