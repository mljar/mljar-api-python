import os
import uuid
import json, requests
import time
import numpy as np
import pandas as pd
from mljar_client import MljarClient

from utils import *

class Error(Exception):
    pass

class IncorrectInputDataError(Error):
    pass

class DatasetUnknownError(Error):
    pass

class DatasetInvalidError(Error):
    pass


class Mljar(MljarClient):

    '''
        This is a wrapper over MLJAR API - it do all the stuff.
    '''

    def __init__(self, project, experiment,
                        metric = '', algorithms = [],
                        preprocessing = [],
                        validation  = MLJAR_DEFAULT_VALIDATION,
                        tuning_mode = MLJAR_DEFAULT_TUNING_MODE,
                        time_constraint = MLJAR_DEFAULT_TIME_CONSTRAINT,
                        create_enseble  = MLJAR_DEFAULT_ENSEMBLE):
        super(Mljar, self).__init__()
        if project == '' or experiment == '':
            raise Exception('The project or experiment title is undefined')

        self.project_title    = project
        self.experiment_title = experiment
        self.validation       = validation
        self.algorithms       = algorithms
        self.preprocessing    = preprocessing
        self.metric           = metric
        self.tuning_mode      = tuning_mode
        self.time_constraint  = time_constraint
        self.create_enseble   = create_enseble


    def _add_project_if_notexists(self, verbose = True):
        '''
            Checks if project exists, if not it add new project.
        '''
        projects = self.get_projects(verbose=False)
        project_details = [p for p in projects if p['title'] == self.project_title]
        # if project with such title does not exist, create one
        if len(project_details) == 0:
            print 'Create a new project:', self.project_title
            project_details = self.create_project(title = self.project_title,
                                        description = 'Porject generated from mljar client API',
                                        task = self.project_task)
        else:
            project_details = project_details[0]
        if verbose:
            print 'Project:', project_details['title']
        return project_details

    def _add_dataset_if_notexists(self, X, y, project_details, verbose = True):
        '''
            Checks if dataset already exists, if not it add dataset to project.
        '''
        data = None
        if isinstance(X, np.ndarray):
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
            data = pd.concat((X,y), axis=1)
            data.columns[-1] = 'target'
        # compute hash to check if dataset already exists
        dataset_hash = str(make_hash(data))
        dataset_details = [d for d in project_details['datasets'] if d['dataset_hash'] == dataset_hash]
        # add new dataset
        if len(dataset_details) == 0:
            self.dataset_title = 'Train-' + str(uuid.uuid4())[:4]
            file_path = '/tmp/dataset-'+ str(uuid.uuid4())[:8]+'.csv'
            data.to_csv(file_path, index=False)
            dataset_details = self.add_new_dataset(project_details['hid'], self.dataset_title, file_path, prediction_only=False)
            print 'New dataset (%s) added to project: %s' % (self.dataset_title, self.project_title)
        else:
            dataset_details = dataset_details[0]
            self.dataset_title = dataset_details.get('title', '')

        # wait till dataset is validated ...
        my_dataset = None
        for i in xrange(60):
            datasets = self.get_datasets(project_hid = project_details['hid'])
            my_dataset = [d for d in datasets if d['title'] == self.dataset_title]
            if len(my_dataset) == 0:
                raise DatasetUnknownError('Can not find dataset: %s' % self.dataset_title)
            my_dataset = my_dataset[0]
            if my_dataset['valid'] == 1:
                break
            time.sleep(10)

        if my_dataset['valid'] != 1:
            raise DatasetInvalidError('Sorry, your dataset can not be understand by MLJAR. Please report this to us - we will fix it.')

        if my_dataset['accepted'] == 0:
            details = self.accept_dataset_column_usage(my_dataset['hid'])
            # and refresh dataset
            datasets = self.get_datasets(project_hid = project_details['hid'])
            my_dataset = [d for d in datasets if d['title'] == self.dataset_title]
            if len(my_dataset) == 0:
                raise Exception('Can not find dataset')
            my_dataset = my_dataset[0]

        if verbose:
            print 'Dataset:', my_dataset['title']
        return my_dataset

    def _add_experiment_if_notexists(self, project_details, dataset_details, verbose = True):
        # get existing experiments
        experiments = self.get_experiments(project_details['hid'])
        experiment_details = [e for e in experiments if e['title'] == self.experiment_title]
        if len(experiment_details) > 0:
            experiment_details = experiment_details[0]
        else:
            dataset_preproc = {}
            # default preprocessing
            if len(dataset_details['column_usage_min']['cols_to_fill_na']) > 0:
                dataset_preproc['na_fill'] = 'na_fill_median'
                print 'There are missing values in dataset which will be filled with median.'
            if len(dataset_details['column_usage_min']['cols_to_convert_categorical']) > 0:
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
                self.algorithms = MLJAR_DEFAULT_ALGORITHMS[project_details['task']]


            params = json.dumps({
                'train_dataset': {'id': dataset_details['hid'], 'title': dataset_details['title']},
                'preproc': dataset_preproc,
                'algs': self.algorithms,
                'ensemble': self.create_enseble,
                'random_start_cnt': MLJAR_TUNING_MODES[self.tuning_mode]['random_start_cnt'],
                'hill_climbing_cnt': MLJAR_TUNING_MODES[self.tuning_mode]['hill_climbing_cnt'],
                'single_limit': self.time_constraint

            })
            data = {
                'title': self.experiment_title,
                'description': 'Auto ...',
                'metric': self.metric,
                'validation_scheme': self.validation,
                'task': project_details['task'],
                'compute_now': 1,
                'parent_project': project_details['hid'],
                'params': params

            }

            experiment_details = elf.create_experiment(data)
        if verbose:
            print 'Experiment:', experiment_details['title'], \
                    'metric:', experiment_details['metric'], \
                    'validation:', experiment_details['validation_scheme']
        return experiment_details

    def _init_experiment(self, X, y):
        # check input data dimensions
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise IncorrectInputDataError('Sorry, multiple outputs are not supported in MLJAR')
        if y.shape[0] != X.shape[0]:
            raise IncorrectInputDataError('Sorry, there is a missmatch between X and y matrices shapes')
        # define project task
        self.project_task = 'bin_class' # binary classification
        if len(np.unique(y)) != 2:
            self.project_task = 'reg' # regression
        #
        # check if project with such title exists
        #
        project_details = self._add_project_if_notexists()
        #
        # add a dataset to project
        #
        dataset_details = self._add_dataset_if_notexists(X, y, project_details)
        #
        # add experiment to project
        #
        experiment_details = self._add_experiment_if_notexists(project_details, dataset_details)
        #
        # get results
        #
        results = self.fetch_results(project_details['hid'], verbose = True)


    def fetch_results(self, project_hid, verbose = False):
        results = self.get_results(project_hid)
        results = [r for r in results if r['experiment'] == self.experiment_title]
        if verbose:
            print "{:{width}} {} \t {} {} [{}]".format('Model', 'Score', 'Metric', 'Validation', 'Status', width=27)
            print '-'* 100
            for r in results:
                model_name = ''
                if r['model_type'] in MLJAR_BIN_CLASS:
                    model_name = MLJAR_BIN_CLASS[r['model_type']]
                if r['model_type'] in MLJAR_REGRESSION:
                    model_name = MLJAR_REGRESSION[r['model_type']]
                if model_name == '':
                    model_name = r['model_type']
                print "{:{width}} {} \t {} {} [{}]".format(model_name, r['metric_value'], r['metric_type'], r['validation_scheme'], r['status'], width=27)


             #{u'status': u'Done', u'metric_value': 0.0, u'iters': 1000, u'model_type': u'xgb',
             #u'status_detail': u'None', u'validation_scheme': u'5fold',
             #u'status_modify_at': u'2016-12-14T10:24:42.005Z', u'experiment': u'Experiment 1', u'hid': u'GvRdeYwVmJak',
             #u'run_time': 3, u'metric_type': u'logloss', u'dataset': u'Train-018a'},


        return results

    def fit(self, X, y):
        self._init_experiment(X, y)



    def predict(self, X):
        print 'MLJAR predict ...'
