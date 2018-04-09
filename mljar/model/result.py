from marshmallow import Schema, fields, post_load

from .base import BaseModel

class ResultSchema(Schema):
    hid = fields.Str()
    experiment = fields.Str()
    dataset = fields.Str()
    validation_scheme = fields.Str()
    model_type = fields.Str()
    metric_type = fields.Str()
    metric_value = fields.Number(allow_none=True)
    run_time = fields.Number(allow_none=True)
    iters = fields.Number(allow_none=True)
    status = fields.Str()
    status_detail = fields.Str(allow_none=True)
    status_modify_at = fields.DateTime()
    importance = fields.Dict(allow_none=True)
    train_prediction_path = fields.Str(allow_none=True)
    params = fields.Dict(allow_none=True)
    train_details = fields.Dict(allow_none=True)
    models_saved = fields.Str(allow_none=True)
    metric_additional = fields.Dict(allow_none=True)

    @post_load
    def make_result_instance(self, data):
        return Result(**data)

class Result(BaseModel):
    schema = ResultSchema(strict=True)

    def __init__(self, hid, experiment, dataset, validation_scheme, model_type, metric_type,
                    params, status, status_detail=None, status_modify_at=None, metric_value=None,
                    importance=None, train_prediction_path=None, run_time=None, iters=None, train_details=None,
                    metric_additional=None, models_saved=None):
        self.hid = hid
        self.experiment = experiment
        self.dataset = dataset
        self.validation_scheme = validation_scheme
        self.model_type = model_type
        self.metric_type = metric_type
        self.metric_value = metric_value
        self.run_time = run_time
        self.iters = iters
        self.status = status
        self.status_detail = status_detail
        self.status_modify_at = status_modify_at
        self.importance = importance
        self.train_prediction_path = train_prediction_path
        self.params = params
        self.train_details = train_details
        self.models_saved = models_saved
        self.metric_additional = metric_additional

    def __str__(self):
        desc = 'Result id: {} model: {} status: {}\n'.format(self.hid, self.model_type, self.status)
        desc += 'Performance: {} on {} with {}\n'.format(str(self.metric_value), self.metric_type, self.validation_scheme)
        return desc

    '''
    def _get_full_model_name(self, model_type):
        model_name = ''
        if model_type in MLJAR_BIN_CLASS:
            model_name = MLJAR_BIN_CLASS[model_type]
        if model_type in MLJAR_REGRESSION:
            model_name = MLJAR_REGRESSION[model_type]
        if model_name == '':
            model_name = model_type
        return model_name
    '''
