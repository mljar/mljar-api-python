import os
import uuid
import json, requests
import numpy as np
from mljar_client import MljarClient


class Mljar(MljarClient):

    def __init__(self, project_title = '', experiment_title = ''):
        super(Mljar, self).__init__()
        if project_title == '':
            project_title = 'Project-' + str(uuid.uuid4())[:4]
        if experiment_title == '':
            experiment_title = 'Experiment-' + str(uuid.uuid4())[:4]
        self.project_title = project_title
        self.experiment_title = experiment_title
        print 'Mljar', self.project_title, self.experiment_title

    def _init_project(self, X, y):
        self.project_task = 'bin_class' # binary classification
        if len(np.unique(y)) != 2:
            self.project_task = 'reg' # regression

        # check if project with such title exists
        projects = self.get_projects(verbose=False)
        project_details = [p for p in projects if p['title'] == self.project_title]
        print project_details
        # if project with such title does not exist, create one
        if len(project_details) == 0:
            print 'Create a new project'
            project_details = self.create_project(title = self.project_title,
                                        description = 'Porject generated from mljar client API',
                                        task = self.project_task)
        else:
            print 'Project already exists'
            project_details = project_details[0]
        print 'Details', project_details
        print project_details['hid']


    def fit(self, X, y):
        print 'MLJAR fit ...'
        #AttributeError: 'Series' object has no attribute 'columns'
        self._init_project(X, y)



    def predict(self, X):
        print 'MLJAR predict ...'
