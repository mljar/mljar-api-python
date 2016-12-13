import os
import uuid
import json, requests
import time
import numpy as np
import pandas as pd
from mljar_client import MljarClient

from utils import *

class Mljar(MljarClient):

    '''
        This is a wrapper over MLJAR API - it do all the stuff.
    '''

    def __init__(self, project = '', experiment = '',
                        metric = '', algorithms = [],
                        preprocessing = [],
                        validation  = MLJAR_DEFAULT_VALIDATION,
                        tuning_mode = MLJAR_DEFAULT_TUNING_MODE,
                        time_constraint = MLJAR_DEFAULT_TIME_CONSTRAINT,
                        create_enseble  = MLJAR_DEFAULT_ENSEMBLE):
        super(Mljar, self).__init__()
        if project == '':
            project = 'Project-' + str(uuid.uuid4())[:4]
        if experiment == '':
            experiment = 'Experiment-' + str(uuid.uuid4())[:4]
        self.project_title    = project
        self.experiment_title = experiment
        self.validation       = validation
        self.algorithms       = algorithms
        self.preprocessing    = preprocessing
        self.metric           = metric
        self.tuning_mode      = tuning_mode
        self.time_constraint  = time_constraint
        self.create_enseble   = create_enseble

        print 'Mljar', self.project_title, self.experiment_title

    def _init_experiment(self, X, y):
        print 'Init experiment'
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise Exception('Sorry, multiple outputs are not supported in MLJAR')
        if y.shape[0] != X.shape[0]:
            raise Exception('Sorry, there is a missmatch between X and y matrices shapes')
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

        # add a dataset to project
        print type(X), type(y)
        data = None
        if isinstance(X, np.ndarray):
            print 'numpy data', X.shape
            cols = {}
            col_names = []
            for i in xrange(X.shape[1]):
                c = 'attribute_'+str(i+1)
                cols[c] = X[:,i]
                col_names += [c]
            cols['target'] = y
            col_names.append('target')
            data = pd.DataFrame(cols, columns=col_names)
        if isinstance(X, pd.DataFrame):
            print 'pandas data'
            data = pd.concat((X,y), axis=1)
            data.columns[-1] = 'target'

        print 'Columns', data.columns

        dataset_hash = str(make_hash(data))
        print 'Hash', dataset_hash

        dataset_details = [d for d in project_details['datasets'] if d['dataset_hash'] == dataset_hash]
        if len(dataset_details) == 0:
            print 'Add new dataset into project'
            self.dataset_title = 'Train-' + str(uuid.uuid4())[:4]
            file_path = '/tmp/dataset-'+ str(uuid.uuid4())[:8]+'.csv'
            data.to_csv(file_path, index=False)
            print file_path
            print "tu", project_details['hid'], self.dataset_title, file_path
            dataset_details = self.add_new_dataset(project_details['hid'], self.dataset_title, file_path, prediction_only=False)

        else:
            print 'Data set already exists!'
            dataset_details = dataset_details[0]
            self.dataset_title = dataset_details.get('title', '')

        print 'There are following datasets in the project', project_details['datasets']
        print 'Dataset details', dataset_details

        # wait till dataset is validated ...
        my_dataset = None
        for i in xrange(60):
            print i, 'Check if dataset is valid ...'

            datasets = self.get_datasets(project_hid = project_details['hid'])
            my_dataset = [d for d in datasets if d['title'] == self.dataset_title]
            if len(my_dataset) == 0:
                raise Exception('Can not find dataset')
            my_dataset = my_dataset[0]
            print 'Valid', my_dataset['valid']
            #print datasets
            if my_dataset['valid'] == 1:
                break
            time.sleep(10)

        if my_dataset['valid'] != 1:
            raise Exception('Sorry, your dataset can not be understand by MLJAR. Please report this to us - we will fix it.')

        if my_dataset['accepted'] == 0:
            print 'Accept dataset ...'
            details = self.accept_dataset_column_usage(my_dataset['hid'])
            print 'Accept details', details
            # and refresh dataset
            datasets = self.get_datasets(project_hid = project_details['hid'])
            my_dataset = [d for d in datasets if d['title'] == self.dataset_title]
            if len(my_dataset) == 0:
                raise Exception('Can not find dataset')
            my_dataset = my_dataset[0]

        print 'MY DATASET', my_dataset



        dataset_hid = my_dataset['hid']
        dataset_title = my_dataset['title']
        dataset_preproc = {}
        # default preprocessing
        if len(my_dataset['column_usage_min']['cols_to_fill_na']) > 0:
            dataset_preproc['na_fill'] = 'na_fill_median'
            print 'There are missing values in dataset which will be filled with median.'
        if len(my_dataset['column_usage_min']['cols_to_convert_categorical']) > 0:
            dataset_preproc['convert_categorical'] = 'categorical_to_int'
            print 'There are categorical attributes which will be coded as integers.'


        # create a new experiment
        if self.validation is None or self.validation == '' or self.validation not in MLJAR_VALIDATIONS:
            self.validation = MLJAR_DEFAULT_VALIDATION
        if self.metric is None or  self.metric == '' or self.metric not in MLJAR_METRICS:
            self.metric = MLJAR_DEFAULT_METRICS[self.project_task]
        if self.tuning_mode is None or  self.tuning_mode == '' or self.tuning_mode not in MLJAR_TUNING_MODES:
            self.tuning_mode = MLJAR_DEFAULT_TUNING_MODE
        if self.algorithms is None or self.algorithms == [] or self.algorithms == '':
            self.algorithms = MLJAR_DEFAULT_ALGORITHMS[self.project_task]

        data = {
            'title': self.experiment_title,
            'description': 'Auto generated ML experiment',
            'metric': self.metric,
            'validation_scheme': self.validation,
            'task': self.project_task,
            'compute_now': 1,
            'parent_project': project_details['hid'],
            'params': str(json.dumps({
                'train_dataset': {'id': dataset_hid, 'title': dataset_title},
                'preproc': dataset_preproc,
                'algs': self.algorithms,
                'ensemble': self.create_enseble,
                'random_start_cnt': MLJAR_TUNING_MODES[self.tuning_mode]['random_start_cnt'],
                'hill_climbing_cnt': MLJAR_TUNING_MODES[self.tuning_mode]['hill_climbing_cnt'],
                'single_limit': self.time_constraint
            }))
        }

        '''

        '''

        print 'Experiment setup', data
        self.create_experiment(data)

    def fit(self, X, y):
        print 'MLJAR fit ...'
        #AttributeError: 'Series' object has no attribute 'columns'
        self._init_experiment(X, y)



    def predict(self, X):
        print 'MLJAR predict ...'
