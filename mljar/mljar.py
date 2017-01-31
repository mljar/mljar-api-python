import os
import uuid
import sys
import json, requests
import time
import numpy as np
import pandas as pd
#from mljar_client import MljarClient

from utils import *
from exceptions import BadValueException, IncorrectInputDataException, UndefinedExperimentException

from client.project import ProjectClient
from client.dataset import DatasetClient
from client.experiment import ExperimentClient
from client.result import ResultClient
from client.prediction import PredictionClient
from client.predictjob import PredictJobClient
from client.prediction_download import PredictionDownloadClient

from log import logger

class Mljar(object):
    '''
    This is a wrapper over MLJAR API - it does all the stuff.
    '''

    def __init__(self, project, experiment,
                        metric = '',
                        algorithms = [],
                        preprocessing = [],
                        validation  = MLJAR_DEFAULT_VALIDATION,
                        tuning_mode = MLJAR_DEFAULT_TUNING_MODE,
                        create_enseble  = MLJAR_DEFAULT_ENSEMBLE,
                        single_algorithm_time_limit = MLJAR_DEFAULT_TIME_CONSTRAINT):
        '''
        Set up MLJAR project and experiment.
        Args:
            tuning_mode: This parameter controls number of models that will be checked
                            for each selected algorithm. There available modes: Normal, Sport, Insane.
            algorithms: The list of algorithms that will be checked. The list depends on project task which will be guessed based on target column values.
                        For binary classification task available algorithm are:
                         - xgb which is for Xgboost
                         - lgb which is for LightGBM
                         - mlp which is for Neural Network
                         - rfc which is for Random Forest
                         - etc which is for Extra Trees
                         - rgfc which is for Regularized Greedy Forest
                         - knnc which is for k-Nearest Neighbors
                         - logreg which is for Logistic Regression
                        For regression task there are available algorithms:
                         - xgbr which is for Xgboost
                         - lgbr which is for LightGBM
                         - rgfr which is for Regularized Greedy Forest
                         - rfr which is for Random Forest
                         - etr which is for Extra Trees
                        You can specify the list of algorithms that you want to use, if left blank all algorithms will be used.
            metric: The metric that will be used for model search and tuning. It depends on project's task.
                    For binary classification there are metrics:
                     - auc which is for Area Under ROC Curve
                     - logloss which is for Logarithmic Loss
                    For regression tasks:
                     - rmse which is Root Mean Square Error
                     - mse which is for Mean Square Error
                     - mase which is for Mean Absolute Error
            validation: The schema of validation that will be used for model search and tuning. There is only available
                        validation with cross validation. Proper values are:
                         - 3fold for 3-fold Stratified CV
                         - 5fold for 5-fold Stratified CV
                         - 10fold for 10-fold Stratified CV
                        The default is 5-fold CV.
            single_algorithm_time_limit: The time in minutes that will be spend for training single algorithm.
                        Default value is 5 minutes.
        '''
        super(Mljar, self).__init__()
        if project == '' or experiment == '':
            raise BadValueException('The project or experiment title is undefined')

        self.project_title    = project
        self.experiment_title = experiment
        self.preprocessing    = preprocessing
        self.create_enseble   = create_enseble
        self.selected_algorithm = None
        self.dataset_title      = None
        self.verbose            = True

        if tuning_mode is None:
            tuning_mode = 'Sport'
        if tuning_mode not in ['Normal', 'Sport', 'Insane']:
            raise BadValueException('There is a wrong tuning mode selected. \
                                        There are available modes: Normal, Sport, Insane.')
        self.tuning_mode = tuning_mode
        # below params are validated later
        self.algorithms = algorithms
        self.metric = metric
        self.validation = validation
        self.single_algorithm_time_limit = single_algorithm_time_limit

    def fit(self, X, y):
        '''
        Fit models with MLJAR engine.
        Args:
            X: The numpy or pandas matrix with training data.
            y: The numpy or pandas vector with target values.
            tuning_mode: This parameter controls number of models that will be checked
                            for each selected algorithm. There available modes: Normal, Sport, Insane.
            algorithms: The list of algorithms that will be checked. The list depends on project task which will be guessed based on target column values.
                        For binary classification task available algorithm are:
                         - xgb which is for Xgboost
                         - lgb which is for LightGBM
                         - mlp which is for Neural Network
                         - rfc which is for Random Forest
                         - etc which is for Extra Trees
                         - rgfc which is for Regularized Greedy Forest
                         - knnc which is for k-Nearest Neighbors
                         - logreg which is for Logistic Regression
                        For regression task there are available algorithms:
                         - xgbr which is for Xgboost
                         - lgbr which is for LightGBM
                         - rgfr which is for Regularized Greedy Forest
                         - rfr which is for Random Forest
                         - etr which is for Extra Trees
                        You can specify the list of algorithms that you want to use, if left blank all algorithms will be used.
            metric: The metric that will be used for model search and tuning. It depends on project's task.
                    For binary classification there are metrics:
                     - auc which is for Area Under ROC Curve
                     - logloss which is for Logarithmic Loss
                    For regression tasks:
                     - rmse which is Root Mean Square Error
                     - mse which is for Mean Square Error
                     - mase which is for Mean Absolute Error
            validation: The schema of validation that will be used for model search and tuning. There is only available
                        validation with cross validation. Proper values are:
                         - 3fold for 3-fold Stratified CV
                         - 5fold for 5-fold Stratified CV
                         - 10fold for 10-fold Stratified CV
                        The default is 5-fold CV.
            single_algorithm_time_limit: The time in minutes that will be spend for training single algorithm.
                        Default value is 5 minutes.
        '''
        # check input data dimensions
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise IncorrectInputDataException('Sorry, multiple outputs are not supported in MLJAR')
        if y.shape[0] != X.shape[0]:
            raise IncorrectInputDataException('Sorry, there is a missmatch between X and y matrices shapes')

        try:
            self.selected_algorithm = self._start_experiment(X, y)
            '''
            if self.selected_algorithm is not None:
                print 'The most useful algotihm:'
                print "{} = {} {} on {} [{}]".format(\
                                            self._get_full_model_name(self.selected_algorithm['model_type']), \
                                            self.selected_algorithm['metric_value'], \
                                            self.selected_algorithm['metric_type'], \
                                            self.selected_algorithm['validation_scheme'], \
                                            self.selected_algorithm['status'])
            '''
        except Exception as e:
            print 'Ups, %s' % str(e)


    def _start_experiment(self, X, y):

        # define project task
        self.project_task = 'bin_class' if len(np.unique(y)) == 2 else 'reg'
        #
        # check if project with such title exists
        #
        self.project = ProjectClient().create_project_if_notexists(self.project_title, self.project_task)
        #
        # add a dataset to project
        #
        self.dataset = DatasetClient(self.project.hid).add_dataset_if_not_exists(X, y)
        #
        # add experiment to project
        #
        self.experiment = ExperimentClient(self.project.hid).add_experiment_if_not_exists(self.dataset, self.experiment_title, self.project_task, \
                                                    self.validation, self.algorithms, self.metric, \
                                                    self.tuning_mode, self.single_algorithm_time_limit, self.create_enseble)
        if self.experiment is None:
            raise UndefinedExperimentException()
        #
        # get results
        #
        results = ResultClient(self.project.hid).get_results(self.experiment.hid)
        #
        # wait for models ...
        #
        the_best_result = self._wait_till_all_models_trained()
        return the_best_result

    def _wait_till_all_models_trained(self):
        WAIT_INTERVAL = 10.0
        loop_max_counter = 60 # 1 hour waiting is enough ;)
        results = None
        while True:
            loop_max_counter -= 1
            if loop_max_counter <= 0:
                break
            try:
                # get current state of the results
                results = ResultClient(self.project.hid).get_results(self.experiment.hid)
                # check if experiment is done, if yes then stop training
                self.experiment = ExperimentClient(self.project.hid).get_experiment(self.experiment.hid)
                if self.experiment.compute_now == 2:
                    break
                # print current state of the results
                initiated_cnt, learning_cnt, done_cnt, error_cnt = self._get_results_stats(results)
                eta = self._asses_total_training_time(results)
                sys.stdout.write("\rinitiated: {}, learning: {}, done: {}, error: {} | ETA: {} minutes                         ".format(initiated_cnt, learning_cnt, done_cnt, error_cnt, eta))
                sys.stdout.flush()

                time.sleep(WAIT_INTERVAL)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error('There is some problem while waiting for models, %s' % str(e))
        logger.info('Get the best result')
        # get the best result!
        self.selected_algorithm = self._get_the_best_result(results)
        return self.selected_algorithm


    def _asses_total_training_time(self, results):
        '''
            Estimated time of models arrival, in minutes.
        '''
        single_alg_limit = float(self.experiment.params.get('single_limit', 5.0))
        initiated_cnt, learning_cnt, done_cnt, error_cnt = self._get_results_stats(results)
        total = (initiated_cnt * single_alg_limit) / float(max(learning_cnt,1))
        total += 0.5 * single_alg_limit
        return total

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

    def _get_the_best_result(self, results):
        the_best_result = None
        if self.experiment.compute_now in [1, 2]:
            opt_direction = 1 if self.experiment.metric \
                                        not in MLJAR_OPT_MAXIMIZE else -1
            min_value = 10e12
            for r in results:
                if r.metric_value is None:
                    continue
                if r.metric_value*opt_direction < min_value:
                    min_value = r.metric_value*opt_direction
                    the_best_result = r
        return the_best_result


    def predict(self, X):
        if self.selected_algorithm is None or self.project is None or self.experiment is None:
            print 'Can not run prediction.'
            print 'Please run fit method first, to fit models and to retrieve them ;)'
            return None
        else:

            # chack if dataset exists in mljar if not upload dataset for prediction
            dataset = DatasetClient(self.project.hid).add_dataset_if_not_exists(X, y = None)

            # check if prediction is available
            total_checks = 100
            for i in xrange(total_checks):
                prediction = PredictionClient(self.project.hid).\
                                get_prediction(dataset.hid, self.selected_algorithm.hid)

                # prediction is not available, first check so submit job
                if i == 0 and prediction is None:
                    # create prediction job
                    submitted = PredictJobClient().submit(self.project.hid, dataset.hid,
                                                            self.selected_algorithm.hid)
                    if not submitted:
                        logger.error('Problem with prediction for your dataset')
                        return None

                if prediction is not None:
                    pred = PredictionDownloadClient().download(prediction.hid)
                    sys.stdout.write('\r\n')
                    return pred

                sys.stdout.write('\rFetch predictions: {0}%'.format(round(i/(total_checks*0.01))))
                sys.stdout.flush()
                time.sleep(5)

            sys.stdout.write('\r\n')
            logger.error('Sorry, there was some problem with computing prediction for your dataset. \
                            Please login to mljar.com to your account and check details.')
            return None
