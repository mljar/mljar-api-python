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

    def __str__(self):
        desc = 'Dataset id: {} title: {} file: {}\n'.format(self.hid, self.title, self.file_name)
        desc += 'File size: {} accepted column usage: {}\n'.format(self.file_size, self.accepted)
        return desc
