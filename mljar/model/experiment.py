from marshmallow import Schema, fields, post_load

from base import BaseModel

class ExperimentSchema(Schema):
    hid = fields.Str()
    title = fields.Str()
    created_at = fields.DateTime(allow_none=True)
    created_by = fields.Number(allow_none=True)
    parent_project = fields.Str(allow_none=True)
    models_cnt = fields.Number()
    task = fields.Str()
    description = fields.Str(allow_none=True)
    metric = fields.Str()
    validation_scheme = fields.Str()
    total_timelog = fields.Str(allow_none=True)
    bestalg = fields.List(fields.Dict(), allow_none=True)
    details = fields.Dict(allow_none=True)
    params = fields.Dict(allow_none=True)
    compute_now = fields.Number()
    computation_started_at = fields.DateTime(allow_none=True)

    @post_load
    def make_experiment_instance(self, data):
        return Experiment(**data)

class Experiment(BaseModel):
    schema = ExperimentSchema(strict=True)

    def __init__(self, hid, title, models_cnt, task, description, metric,
                    validation_scheme, details,
                    params, compute_now, computation_started_at,
                    bestalg = None, total_timelog = None,
                    created_at = None, created_by = None, parent_project = None):
        self.hid = hid
        self.title = title
        self.description = description
        self.created_at = created_at
        self.created_by = created_by
        self.parent_project = parent_project
        self.models_cnt = models_cnt
        self.task = task
        self.metric = metric
        self.validation_scheme = validation_scheme
        self.total_timelog = total_timelog
        self.bestalg = bestalg
        self.details = details
        self.params = params
        self.compute_now = compute_now
        self.computation_started_at = computation_started_at

    def show(self):
        print '-'*50,'\Experiment details (', self.hid,')\n','-'*50
        print 'Title:', self.title
        print 'Metric:', self.metric
        print 'Validation:', self.validation_scheme
        print 'Algorithms:', self.params.get('algs', None)
        print 'Single algorithm train time', self.params.get('single_limit', None), 'minutes'
        print '-'*50

    def equal(self, expt):
        # sort algorithms names before comparison
        algs   = sorted(self.params.get('algs', []))
        algs_2 = sorted(expt.params.get('algs', []))
        return  self.params['train_dataset'].get('hid', None) == expt.params['train_dataset'].get('hid', None) and \
                self.metric == str(expt.metric) and \
                self.validation_scheme == str(expt.validation_scheme) and \
                '-'.join(algs) == '-'.join(algs_2) and \
                int(self.params.get('single_limit', 0)) == int(expt.params.get('single_limit', 0)) and \
                self.params.get('preproc', None) == expt.params.get('preproc', None)
