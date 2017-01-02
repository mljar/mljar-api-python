import os
import uuid
import sys
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
        self.selected_algorithm = None
        self.dataset_title      = None

    def _add_project_if_notexists(self, verbose = True):
        '''
            Checks if project exists, if not it add new project.
        '''
        projects = self.get_projects(verbose=False)
        self.project_details = [p for p in projects if p['title'] == self.project_title]
        # if project with such title does not exist, create one
        if len(self.project_details) == 0:
            print 'Create a new project:', self.project_title
            self.project_details = self.create_project(title = self.project_title,
                                        description = 'Porject generated from mljar client API',
                                        task = self.project_task)
        else:
            self.project_details = self.project_details[0]
        if verbose:
            print 'Project:', self.project_details['title']
        return self.project_details

    def _wait_till_all_datasets_are_valid(self, dataset_hash):
        #datasets = self.project_details['datasets']
        # try to refresh project details
        #if len(datasets) == 0:
        self.project_details = self.get_project_details(self.project_details['hid'])
        datasets = self.project_details['datasets']

        if len(datasets) == 0:
            return None


        not_validated = [ds for ds in datasets if ds['valid'] == 0]
        if len(not_validated) > 0:
            print 'MLJAR is computing statistics for your dataset.'
            print 'When ready, you can go to you project: %s' % self.project_details['title']
            print 'and view data statistics in Preview. Please wait a moment.'

            my_dataset = None
            total_checks = 120
            done = False
            for i in xrange(120):
                sys.stdout.write('\rProgress: {0}%'.format(round(i/(total_checks*0.01))))
                sys.stdout.flush()
                datasets = self.get_datasets(project_hid = self.project_details['hid'])
                not_validated = [ds for ds in datasets if ds['valid'] == 0]
                if len(not_validated) == 0:
                    if self.dataset_title is not None:
                        my_dataset = [d for d in datasets if d['title'] == self.dataset_title]
                        my_dataset = my_dataset[0] if len(my_dataset) > 0 else None
                    done = True
                    break
                time.sleep(5)
            if done:
                print '\rStatistics are available at https://mljar.com/app/#/p/%s/datapreview' % self.project_details['hid']
            else:
                print '\rSorry, the statistics for dataset can not be computed right now. Please try again in a while.'

        else:
            # all valid
            my_dataset = [d for d in datasets if d['dataset_hash'] == dataset_hash]
            my_dataset = my_dataset[0] if len(my_dataset) > 0 else None

        return my_dataset

    def _add_dataset_if_notexists(self, X, y, verbose = True):
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
            if y is not None:
                cols['target'] = y
                col_names.append('target')
            data = pd.DataFrame(cols, columns=col_names)
        if isinstance(X, pd.DataFrame):
            if y is not None:
                data = pd.concat((X,y), axis=1)
                data.rename({data.columns[-1]:'target'}, inplace=True)
            else:
                data = X

        # compute hash to check if dataset already exists
        dataset_hash = str(make_hash(data))
        dataset_details = [d for d in self.project_details['datasets'] if d['dataset_hash'] == dataset_hash]

        # wait till all dataset are validated
        _ = self._wait_till_all_datasets_are_valid(dataset_hash)


        # add new dataset
        if len(dataset_details) == 0:
            self.dataset_title = 'Dataset-' + str(uuid.uuid4())[:4]
            file_path = '/tmp/dataset-'+ str(uuid.uuid4())[:8]+'.csv'
            data.to_csv(file_path, index=False)
            dataset_details = self.add_new_dataset(self.project_details['hid'], self.dataset_title, file_path, prediction_only=(y is None))
            print 'New dataset (%s) added to project: %s' % (self.dataset_title, self.project_title)
        else:
            dataset_details = dataset_details[0]
            self.dataset_title = dataset_details.get('title', '')

        # wait till dataset is validated ...
        my_dataset = self._wait_till_all_datasets_are_valid(dataset_hash)
        if my_dataset is None or len(my_dataset) == 0:
            raise DatasetUnknownError('Can not find dataset: %s' % self.dataset_title)


        if my_dataset['valid'] != 1:
            raise DatasetInvalidError('Sorry, your dataset can not be understand by MLJAR. Please report this to us - we will fix it.')

        if my_dataset['accepted'] == 0:
            details = self.accept_dataset_column_usage(my_dataset['hid'])
            # and refresh dataset
            datasets = self.get_datasets(project_hid = self.project_details['hid'])
            my_dataset = [d for d in datasets if d['title'] == self.dataset_title]
            if len(my_dataset) == 0:
                raise Exception('Can not find dataset')
            my_dataset = my_dataset[0]

        if verbose:
            print 'Dataset:', my_dataset['title']
        return my_dataset

    def _add_experiment_if_notexists(self, dataset_details, verbose = True):
        # get existing experiments
        experiments = self.get_experiments(self.project_details['hid'])
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
                self.algorithms = MLJAR_DEFAULT_ALGORITHMS[self.project_details['task']]


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
                'task': self.project_details['task'],
                'compute_now': 1,
                'parent_project': self.project_details['hid'],
                'params': params

            }

            experiment_details = self.create_experiment(data)
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
        self.project_details = self._add_project_if_notexists()
        #
        # add a dataset to project
        #
        dataset_details = self._add_dataset_if_notexists(X, y)
        #
        # add experiment to project
        #
        experiment_details = self._add_experiment_if_notexists(dataset_details)
        #
        # get results
        #
        results = self.fetch_results(self.project_details['hid'], verbose = False)

        #print "expt details", experiment_details

        experiment_state = 'Ready for computation'
        if experiment_details['compute_now'] == 1:
            experiment_state = 'Computing'
        if experiment_details['compute_now'] == 2:
            experiment_state = 'Done'
        print 'Experiment\'s state is:', experiment_state

        if experiment_state != 'Done':
            WAIT_INTERVAL = 10.0
            start_eta = int(self._asses_total_training_time(experiment_details, results) * 60.0 / WAIT_INTERVAL)

            print "Models in training:"
            for i in range(start_eta):
                results = self.fetch_results(self.project_details['hid'], verbose = False)
                initiated_cnt, learning_cnt, done_cnt, error_cnt = self._get_results_stats(results)

                eta = self._asses_total_training_time(experiment_details, results)

                sys.stdout.write("\rinitiated: {}, learning: {}, done: {}, error: {} | ETA: {} minutes                         ".format(initiated_cnt, learning_cnt, done_cnt, error_cnt, eta))
                sys.stdout.flush()

                if initiated_cnt + learning_cnt == 0:
                    # get experiment info
                    experiments = self.get_experiments(self.project_details['hid'])
                    experiment_details = [e for e in experiments if e['title'] == self.experiment_title]
                    experiment_details = experiment_details[0] if len(experiment_details) > 0 else None
                    if experiment_details is not None:
                        if experiment_details['compute_now'] == 2: # experiment finished
                            results = self.fetch_results(self.project_details['hid'], verbose = True)
                            experiment_state = 'Done'
                            break
                time.sleep(WAIT_INTERVAL)

        # get the best result!
        the_best_result = None
        if experiment_state in ['Computing', 'Done']:
            opt_direction = 1 if experiment_details['metric'] \
                                        not in MLJAR_OPT_MAXIMIZE else -1
            min_value = 1000000000
            for r in results:
                if r['metric_value']*opt_direction < min_value:
                    min_value = r['metric_value']*opt_direction
                    the_best_result = r
        return the_best_result

    def _get_results_stats(self, results):
        initiated_cnt, learning_cnt, done_cnt, error_cnt = 0, 0, 0, 0
        for r in results:
            if r['status'] == 'Initiated':
                initiated_cnt += 1
            elif r['status'] == 'Learning':
                learning_cnt += 1
            elif r['status'] == 'Done':
                done_cnt += 1
            else: # error
                error_cnt += 1
        return initiated_cnt, learning_cnt, done_cnt, error_cnt

    def _asses_total_training_time(self, experiment_details, results):
        '''
            Estimated time of models arrival, in minutes.
        '''
        single_alg_limit = float(experiment_details['params']['single_limit'])
        initiated_cnt, learning_cnt, done_cnt, error_cnt = self._get_results_stats(results)
        total = (initiated_cnt * single_alg_limit) / float(max(learning_cnt,1))
        total += 0.5 * single_alg_limit
        return total

    def _wait_till_all_models_trained(self, project_details):
        pass

    def _get_full_model_name(self, model_type):
        model_name = ''
        if model_type in MLJAR_BIN_CLASS:
            model_name = MLJAR_BIN_CLASS[model_type]
        if model_type in MLJAR_REGRESSION:
            model_name = MLJAR_REGRESSION[model_type]
        if model_name == '':
            model_name = model_type
        return model_name

    def fetch_results(self, project_hid, verbose = False):
        results = self.get_results(project_hid)
        results = [r for r in results if r['experiment'] == self.experiment_title]
        if verbose:
            print '-'* 80
            print "{:{width}} {} \t {} {} [{}]".format('Model', 'Score', 'Metric', 'Validation', 'Status', width=27)
            print '-'* 80
            for r in results:
                model_name = self._get_full_model_name(r['model_type'])
                print "{:{width}} {} \t {} {} [{}]".format(model_name, r['metric_value'], r['metric_type'], r['validation_scheme'], r['status'], width=27)

        return results

    def fit(self, X, y):
        try:
            self.selected_algorithm = self._init_experiment(X, y)
            if self.selected_algorithm is not None:
                print 'The most useful algotihm:'
                print "{} = {} {} on {} [{}]".format(\
                                            self._get_full_model_name(self.selected_algorithm['model_type']), \
                                            self.selected_algorithm['metric_value'], \
                                            self.selected_algorithm['metric_type'], \
                                            self.selected_algorithm['validation_scheme'], \
                                            self.selected_algorithm['status'])

        except Exception as e:
            print 'Ups, %s' % str(e)



    def predict(self, X):
        if self.selected_algorithm is None or self.project_details is None:
            print 'Can not run prediction.'
            print 'Please run fit first to find algorithm.'
            return None
        else:
            print 'Start prediction'

            # upload dataset for prediction
            dataset_details = self._add_dataset_if_notexists(X, y = None, verbose = True)
            print 'Dataset', dataset_details
            # create prediction job
            job_details = self.submit_predict_job(self.project_details['hid'],
                                                    dataset_details['hid'],
                                                    self.selected_algorithm['hid'])

            # wait for prediction
