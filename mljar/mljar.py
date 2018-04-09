from __future__ import print_function
import os
import uuid
import sys
import json, requests
import time
import numpy as np

from .utils import *
from .exceptions import IncorrectInputDataException, UndefinedExperimentException
from .exceptions import MljarException, BadValueException

from .client.project import ProjectClient
from .client.dataset import DatasetClient
from .client.experiment import ExperimentClient
from .client.result import ResultClient
from .client.prediction import PredictionClient
from .client.predictjob import PredictJobClient
from .client.prediction_download import PredictionDownloadClient

from .log import logger

class Mljar(object):
    '''
    This is a wrapper over MLJAR API - it does all the stuff.
    '''

    def __init__(self, project,
                        experiment,
                        metric = '',
                        algorithms = [],
                        validation_kfolds = MLJAR_DEFAULT_FOLDS,
                        validation_shuffle = MLJAR_DEFAULT_SHUFFLE,
                        validation_stratify = MLJAR_DEFAULT_STRATIFY,
                        validation_train_split = MLJAR_DEFAULT_TRAIN_SPLIT,
                        tuning_mode = MLJAR_DEFAULT_TUNING_MODE,
                        create_ensemble  = MLJAR_DEFAULT_ENSEMBLE,
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
            validation_kfolds: The number of folds to be used in validation,
                            it is omitted if validation_train_split is not None
                            or there is validation dataset provided.
                            It can be number from 2 to 15.
            validation_shuffle: The boolean which specify if shuffle samples before training.
                            It is used in k-fold CV and in validation split. Default is set True.
                            It is ignored when validating with separate dataset.
            validation_stratify: The boolean which decides whether samples will be
                            divided into folds with the same class distribution.
                            In regression tasks this flag is ignored. Default is set to True.
            validation_train_split: The ratio how to split training dataset into train and validation.
                            This ratio specify what ratio from input data should be used in training.
                            It should be from (0.05,0.95) range. If it is not None, then
                            validation_kfolds variable is ignored.
            single_algorithm_time_limit: The time in minutes that will be spend for training single algorithm.
                        Default value is 5 minutes.
        '''
        super(Mljar, self).__init__()
        if project == '' or experiment == '':
            raise BadValueException('The project or experiment title is undefined')

        self.project_title    = project
        self.experiment_title = experiment
        self.create_ensemble   = create_ensemble
        self.selected_algorithm = None
        self.dataset_title      = None
        self.verbose            = True

        if tuning_mode is None:
            tuning_mode = MLJAR_DEFAULT_TUNING_MODE
        if tuning_mode not in ['Normal', 'Sport', 'Insane']:
            raise BadValueException('There is a wrong tuning mode selected. \
                                        There are available modes: Normal, Sport, Insane.')
        self.tuning_mode = tuning_mode
        # below params are validated later
        self.algorithms = algorithms
        self.metric = metric
        self.single_algorithm_time_limit = single_algorithm_time_limit
        self.wait_till_all_done = True
        self.selected_algorithm = None
        self.project = None
        self.experiment = None

        self.validation_kfolds = validation_kfolds
        self.validation_shuffle = validation_shuffle
        self.validation_stratify = validation_stratify
        self.validation_train_split = validation_train_split

        if self.validation_kfolds is not None:
            if self.validation_kfolds < 2 or self.validation_kfolds > 15:
                raise MljarException('Wrong validation_kfolds parameter value, it should be in [2, 15] range.')

        if self.validation_train_split is not None:
            if self.validation_train_split < 0.05 or self.validation_train_split > 0.95:
                raise MljarException('Wrong validation_train_split parameter value, it should be in (0.05, 0.95) range.')


    def fit(self, X, y, validation_data = None, wait_till_all_done = True, dataset_title = None):
        '''
        Fit models with MLJAR engine.
        Args:
            X: The numpy or pandas matrix with training data.
            y: The numpy or pandas vector with target values.
            validation_data: Tuple (X,y) with validation data.If set to None, then
                                the k-fold CV or train split validation will be used.
            wait_till_all_done: The flag which decides if fit function will wait
                                till experiment is done.
            dataset_title: The title of your dataset. It is optional. If missing the
                            random title will be generated.
        '''
        self.wait_till_all_done = wait_till_all_done
        # check input data dimensions
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise IncorrectInputDataException('Sorry, multiple outputs are not supported in MLJAR')
        if y.shape[0] != X.shape[0]:
            raise IncorrectInputDataException('Sorry, there is a missmatch between X and y matrices shapes')

        try:
            self._start_experiment(X, y, validation_data, dataset_title)
        except Exception as e:
            print('Ups, {0}'.format(str(e)))


    def _start_experiment(self, X, y, validation_data = None, dataset_title = None):

        # define project task
        self.project_task = 'bin_class' if len(np.unique(y)) == 2 else 'reg'
        #
        # check if project with such title exists
        #
        logger.info('MLJAR: add project')
        self.project = ProjectClient().create_project_if_not_exists(self.project_title, self.project_task)
        #
        # add a dataset to project
        #
        logger.info('MLJAR: add training dataset')
        self.dataset = DatasetClient(self.project.hid).add_dataset_if_not_exists(X, y, title_prefix = 'Training-', dataset_title = dataset_title)

        self.dataset_vald = None
        if validation_data is not None:
            if len(validation_data) != 2:
                raise MljarException('Wrong format of validation data. It should be tuple (X,y)')
            logger.info('MLJAR: add validation dataset')
            X_vald, y_vald = validation_data
            self.dataset_vald = DatasetClient(self.project.hid).add_dataset_if_not_exists(X_vald, y_vald, title_prefix = 'Validation-')
        #
        # add experiment to project
        #
        logger.info('MLJAR: add experiment')
        self.experiment = ExperimentClient(self.project.hid).add_experiment_if_not_exists(self.dataset, self.dataset_vald, \
                                                    self.experiment_title, self.project_task, \
                                                    self.validation_kfolds, self.validation_shuffle, \
                                                    self.validation_stratify, self.validation_train_split, \
                                                    self.algorithms, self.metric, \
                                                    self.tuning_mode, self.single_algorithm_time_limit, self.create_ensemble)
        if self.experiment is None:
            raise UndefinedExperimentException()
        #
        # get results
        #
        # results = ResultClient(self.project.hid).get_results(self.experiment.hid)
        #
        # wait for models ...
        #
        if self.wait_till_all_done:
            self.selected_algorithm = self._wait_till_all_models_trained()

    def _wait_till_all_models_trained(self):
        WAIT_INTERVAL = 10.0
        loop_max_counter = 24*360 # 24 hours of max waiting, is enough ;)
        results = None
        max_error_cnt = 5
        current_error_cnt = 0
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
                if initiated_cnt + learning_cnt + done_cnt + error_cnt == 0:
                    eta = 'estimating'
                else:
                    eta = round(eta, 2)
                sys.stdout.write("\rinitiated: {}, learning: {}, done: {}, error: {} | ETA: {} minutes                         ".format(initiated_cnt, learning_cnt, done_cnt, error_cnt, eta))
                sys.stdout.flush()

                time.sleep(WAIT_INTERVAL)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error('There is some problem while waiting for models, %s' % str(e))
                current_error_cnt += 1
                if current_error_cnt >= max_error_cnt:
                    break
        logger.info('Get the best result')
        print('') # add new line
        # get the best result!
        return self._get_the_best_result(results)



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
            if r.status == 'Initiated':
                initiated_cnt += 1
            elif r.status == 'Learning':
                learning_cnt += 1
            elif r.status == 'Done':
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
        if self.project is None or self.experiment is None:
            print('Can not run prediction.')
            print('Please run fit method first, to start models training and to retrieve them ;)')
            return None
        if self.selected_algorithm is None:
            results = ResultClient(self.project.hid).get_results(self.experiment.hid)
            self.selected_algorithm = self._get_the_best_result(results)
            if self.experiment.compute_now != 2:
                if self.selected_algorithm is not None:
                    print('DISCLAIMER:')
                    print('Your experiment is not yet finished.')
                    print('You will use the best model up to now.')
                    print('You can obtain better results if you wait till experiment is finished.')
                else:
                    print('There is no ready model to use for prediction.')
                    print('Please wait and try in a moment')
                    return None

        if self.selected_algorithm is not None:

            return Mljar.compute_prediction(X, self.selected_algorithm.hid, self.project.hid)
            '''
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
                    #sys.stdout.write('\r\n')
                    return pred

                #sys.stdout.write('\rFetch predictions: {0}%'.format(round(i/(total_checks*0.01))))
                #sys.stdout.flush()
                time.sleep(5)

            #sys.stdout.write('\r\n')
            logger.error('Sorry, there was some problem with computing prediction for your dataset. \
                            Please login to mljar.com to your account and check details.')
            return None
            '''


    @staticmethod
    def compute_prediction(X, model_id, project_id, keep_dataset = False, dataset_title = None):


        # chack if dataset exists in mljar if not upload dataset for prediction
        dataset = DatasetClient(project_id).add_dataset_if_not_exists(X, y = None, title_prefix = 'Testing-', dataset_title = dataset_title)

        # check if prediction is available
        total_checks = 1000
        for i in range(total_checks):
            prediction = PredictionClient(project_id).\
                            get_prediction(dataset.hid, model_id)

            # prediction is not available, first check so submit job
            if i == 0 and prediction is None:
                # create prediction job
                submitted = PredictJobClient().submit(project_id, dataset.hid,
                                                        model_id)
                if not submitted:
                    logger.error('Problem with prediction for your dataset')
                    return None

            if prediction is not None:
                pred = PredictionDownloadClient().download(prediction.hid)
                #sys.stdout.write('\r\n')
                if not keep_dataset:
                    DatasetClient(project_id).delete_dataset(dataset.hid)
                return pred

            #sys.stdout.write('\rFetch predictions: {0}%'.format(round(i/(total_checks*0.01))))
            #sys.stdout.flush()
            time.sleep(10)

        #sys.stdout.write('\r\n')
        logger.error('Sorry, there was some problem with computing prediction for your dataset. \
                        Please login to mljar.com to your account and check details.')
        return None
