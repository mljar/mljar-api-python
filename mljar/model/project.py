from marshmallow import Schema, fields, post_load

from base import BaseModel

from ..exceptions import UnknownProjectTask

class ProjectSchema(Schema):
    hid = fields.Str()
    title = fields.Str()
    description = fields.Str(allow_none=True)
    task = fields.Str()
    hardware = fields.Str()
    scope = fields.Str()
    info = fields.Dict(allow_none=True)
    created_at = fields.DateTime()
    created_by = fields.Number()
    experiments_cnt = fields.Number()
    models_cnt = fields.Number()
    datasets = fields.List(fields.Dict(), allow_none=True)
    topalg = fields.List(fields.Dict(), allow_none=True)
    total_timelog = fields.Number(allow_none=True)
    compute_now = fields.Number()
    insights = fields.List(fields.Dict(), allow_none=True)

    @post_load
    def make_project_instance(self, data):
        return Project(**data)

class Project(BaseModel):
    schema = ProjectSchema(strict=True)

    def __init__(self, hid, title, description, task, hardware, scope, info, created_at, created_by,
                    experiments_cnt, models_cnt, datasets, topalg,
                    compute_now, insights, total_timelog = 0):
        self.hid = hid
        self.title = title
        self.description = description
        self.task = task
        self.info = info
        self.created_at = created_at
        self.created_by = created_by
        self.experiments_cnt = experiments_cnt
        self.models_cnt = models_cnt
        self.hardware = hardware
        self.scope = scope
        self.datasets = datasets
        self.topalg = topalg
        self.total_timelog = total_timelog
        self.compute_now = compute_now
        self.insights = insights


    def show(self):
        print '-'*50,'\nProject details (', self.hid,')\n','-'*50
        print 'Title:', self.title
        if self.description:
            print 'Description:', self.description
        print 'Task:', self.task
        print 'Hardware:', self.hardware
        print 'User data sources count:', len(self.datasets)
        print 'Models count:', self.models_cnt
        print '-'*50


    def _task_to_full_name(self, task_short):
        tasks = {'bin_class': "Binary classification",
                    'reg': "Regression",
                    'img_class': "Images classification"}
        if task_short not in tasks:
            raise UnknownProjectTask('Unknown task %s' % task_short)
        return tasks[task_short]
