from marshmallow import Schema, fields, post_load

from base import BaseModel

class PredictionSchema(Schema):
    hid = fields.Str()
    scope = fields.Str()
    created_by = fields.Number()
    created_at = fields.DateTime()
    parent_alg_hid = fields.Str()
    prediction_on_dataset_title = fields.Str()
    alg_name = fields.Str()
    alg_on_dataset_title = fields.Str()
    alg_metric = fields.Str()

    @post_load
    def make_prediction_instance(self, data):
        return Prediction(**data)

class Prediction(BaseModel):
    schema = PredictionSchema(strict=True)

    def __init__(self, hid, scope, created_by, created_at, parent_alg_hid,
                    prediction_on_dataset_title, alg_name, alg_on_dataset_title,
                    alg_metric):
        self.hid = hid
        self.scope = scope
        self.created_by = created_by
        self.created_at = created_at
        self.parent_alg_hid = parent_alg_hid
        self.prediction_on_dataset_title = prediction_on_dataset_title
        self.alg_name = alg_name
        self.alg_on_dataset_title = alg_on_dataset_title
        self.alg_metric = alg_metric

    def __str__(self):
        desc = 'Prediction id: {}\n'.format(self.hid)
        return desc
