'''
Mljar tests.
'''
import os
import pandas as pd
import numpy as np
import unittest
import time

from mljar.client.project import ProjectClient
from mljar.client.dataset import DatasetClient
from .project_based_test import ProjectBasedTest
from mljar.exceptions import BadValueException, IncorrectInputDataException
from mljar.utils import MLJAR_DEFAULT_TUNING_MODE
from mljar import Mljar

class MljarTest(ProjectBasedTest):

    def setUp(self):
        self.proj_title = 'Test project-01'+get_postfix()
        self.proj_task = 'bin_class'
        self.expt_title = 'Test expt 1'
        # load data
        df = pd.read_csv('tests/data/test_1.csv')
        cols = ['sepal length', 'sepal width', 'petal length', 'petal width']
        target = 'class'
        self.X = df[cols]
        self.y = df[target]

    def tearDown(self):
        # clean
        ProjectBasedTest.clean_projects()


    def mse(self, predictions, targets):
        predictions = np.array(predictions)
        targets = np.array(targets)
        targets = targets.reshape((targets.shape[0],1))
        return ((predictions - targets) ** 2).mean()


    def test_compute_prediction(self):
        model = Mljar(project = self.proj_title, experiment = self.expt_title,
                        algorithms = ['rfc'], metric = 'logloss',
                        validation_kfolds = 3, tuning_mode = 'Normal',
                        single_algorithm_time_limit = 1)
        self.assertTrue(model is not None)
        # fit models and wait till all models are trained
        model.fit(X = self.X, y = self.y, dataset_title = 'My dataset')

        # get project id
        project_id = model.project.hid
        # get model id
        model_id = model.selected_algorithm.hid

        dc = DatasetClient(project_id)
        init_datasets_cnt = len(dc.get_datasets())
        # compute predictions
        pred = Mljar.compute_prediction(self.X, model_id, project_id)
        # compute score
        score = self.mse(pred, self.y)
        self.assertTrue(score < 0.1)
        # check if dataset was removed
        self.assertEqual(init_datasets_cnt, len(dc.get_datasets()))
        # run predictions again, but keep dataset
        pred = Mljar.compute_prediction(self.X, model_id, project_id, keep_dataset = True)
        self.assertEqual(init_datasets_cnt+1, len(dc.get_datasets())) # should be one more


    def test_basic_usage(self):
        '''
        Test the most common usage.
        '''
        model = Mljar(project = self.proj_title, experiment = self.expt_title,
                        algorithms = ['xgb'], metric = 'logloss',
                        validation_kfolds = 3, tuning_mode = 'Normal',
                        single_algorithm_time_limit = 1)
        self.assertTrue(model is not None)
        # fit models and wait till all models are trained
        model.fit(X = self.X, y = self.y)
        # run prediction
        pred = model.predict(self.X)
        # get MSE
        score = self.mse(pred, self.y)
        self.assertTrue(score < 0.1)

    def test_usage_with_defaults(self):
        '''
        Test usage with defaults.
        '''
        model = Mljar(project = self.proj_title, experiment = self.expt_title)
        self.assertTrue(model is not None)
        # fit models and wait till all models are trained
        model.fit(X = self.X, y = self.y, wait_till_all_done = False)
        # wait some time
        time.sleep(120) # wait a little longer - there are a lot of models
        # run prediction
        pred = model.predict(self.X)
        # get MSE
        score = self.mse(pred, self.y)
        self.assertTrue(score < 0.5)
        # check default validation
        self.assertEqual(model.selected_algorithm.validation_scheme, "5-fold CV, Shuffle, Stratify")

    def test_usage_with_train_split(self):
        '''
        Test usage with train split.
        '''
        model = Mljar(project = self.proj_title, experiment = self.expt_title,
                    validation_train_split = 0.8, algorithms = ['xgb'], tuning_mode='Normal',
                    single_algorithm_time_limit=1)
        self.assertTrue(model is not None)
        # fit models and wait till all models are trained
        model.fit(X = self.X, y = self.y, wait_till_all_done = False)
        # wait some time
        time.sleep(60)
        # run prediction
        pred = model.predict(self.X)
        # get MSE
        score = self.mse(pred, self.y)
        self.assertTrue(score < 0.5)
        # check default validation
        self.assertEqual(model.selected_algorithm.validation_scheme, "Split 80/20, Shuffle, Stratify")


    def test_usage_with_validation_dataset(self):
        '''
        Test usage with validation dataset.
        '''
        model = Mljar(project = self.proj_title, experiment = self.expt_title,
                            algorithms = ['xgb'], tuning_mode='Normal',
                            single_algorithm_time_limit = 1)
        self.assertTrue(model is not None)
        # load validation data
        df = pd.read_csv('tests/data/test_1_vald.csv')
        cols = ['sepal length', 'sepal width', 'petal length', 'petal width']
        target = 'class'
        X_vald = df[cols]
        y_vald = df[target]
        # fit models and wait till all models are trained
        model.fit(X = self.X, y = self.y, validation_data=(X_vald, y_vald), wait_till_all_done = False)
        # wait some time
        time.sleep(80)
        # run prediction
        pred = model.predict(self.X)
        # get MSE
        score = self.mse(pred, self.y)
        self.assertTrue(score < 0.5)
        # check default validation
        self.assertEqual(model.selected_algorithm.validation_scheme, "With dataset")




    def test_empty_project_title(self):
        with self.assertRaises(BadValueException) as context:
            model = Mljar(project = '', experiment = '')

    def test_wrong_tuning_mode(self):
        with self.assertRaises(BadValueException) as context:
            model = Mljar(project = self.proj_title, experiment = self.expt_title,
                            tuning_mode = 'Crazy')

    def test_default_tuning_mode(self):
        model = Mljar(project = self.proj_title, experiment = self.expt_title)
        self.assertEqual(model.tuning_mode, MLJAR_DEFAULT_TUNING_MODE)

    def test_wrong_input_dim(self):
        with self.assertRaises(IncorrectInputDataException) as context:
            model = Mljar(project = self.proj_title, experiment = self.expt_title)
            samples = 100
            columns = 10
            X = np.random.rand(samples, columns)
            y = np.random.choice([0,1], samples+1, replace = True)
            model.fit(X, y)

    def test_predict_without_fit(self):
        """ Call predict without calling first fit method should return None """
        model = Mljar(project = self.proj_title, experiment = self.expt_title)
        pred = model.predict(self.X)
        self.assertTrue(pred is None)

    def test_non_wait_fit(self):
        '''
        Test the non wait fit.
        '''
        model = Mljar(project = self.proj_title, experiment = self.expt_title,
                        algorithms = ['xgb'], metric='logloss',
                        validation_kfolds=3, tuning_mode='Normal',
                        single_algorithm_time_limit = 1)
        self.assertTrue(model is not None)
        # fit models, just start computation and do not wait
        start_time = time.time()
        model.fit(X = self.X, y = self.y, wait_till_all_done = False)
        end_time = time.time()
        # time to initialize models should not be greater than 5 minutes
        self.assertTrue(end_time - start_time < 5*60)
        # run prediction
        # good model is not guaranteed
        # but there should be at least one
        max_trys = 50
        pred = None
        while True:
            pred = model.predict(self.X)
            if pred is None:
                # there is no model ready, please wait
                time.sleep(10)
            else:
                break
            max_trys -= 1
            if max_trys <= 0:
                break

        self.assertTrue(pred is not None)
        # get MSE
        score = self.mse(pred, self.y)
        self.assertTrue(score < 0.99)

    def test_retrive_models(self):
        '''
        Test scenario, when user create project, fit models, and try to once
        again run project. In this case, there will be no additional computations,
        all models will be simply retrived from existing project.
        '''
        model = Mljar(project = self.proj_title, experiment = self.expt_title,
                        algorithms = ['xgb'], metric = 'logloss',
                        validation_kfolds = 3, tuning_mode = 'Normal',
                        single_algorithm_time_limit = 1)
        self.assertTrue(model is not None)
        # fit models and wait till all models are trained
        model.fit(X = self.X, y = self.y)
        # run prediction
        pred = model.predict(self.X)
        # get MSE
        score = self.mse(pred, self.y)
        self.assertTrue(score < 0.1)

        # re-use already trained models
        # call fit but models are already trained
        # should be retrived - this should not be longer than 3 minutes
        start_time = time.time()
        model.fit(X = self.X, y = self.y)
        end_time = time.time()
        self.assertTrue(end_time - start_time < 3*60)
        # check prediction
        pred = model.predict(self.X)
        # get MSE
        score_2 = self.mse(pred, self.y)
        self.assertTrue(score_2 < 0.1)
        # scores should be the same
        self.assertTrue(np.abs(score-score_2) < 1e-3)

        # re-use project
        start_time = time.time()
        model_2 = Mljar(project = self.proj_title, experiment = self.expt_title,
                        algorithms = ['xgb'], metric = 'logloss',
                        validation_kfolds = 3, tuning_mode = 'Normal',
                        single_algorithm_time_limit = 1)
        self.assertTrue(model_2 is not None)
        # re-use trained models
        model_2.fit(X = self.X, y = self.y)
        end_time = time.time()
        # it should not take longer than 5 minutes
        self.assertTrue(end_time - start_time < 5*60)
        # run prediction
        pred = model_2.predict(self.X)
        # get MSE
        score_3 = self.mse(pred, self.y)
        self.assertTrue(score_3 < 0.1)
        # scores should be the same
        self.assertTrue(np.abs(score-score_3) < 1e-3)

    '''
    # comment out because it took too long on travis-ci to run
    def test_basic_usage_with_defaults(self):

        #Test the most common usage with defults settings.

        model = Mljar(project = self.proj_title, experiment = self.expt_title)
        self.assertNotEqual(model, None)
        # fit models and wait till all models are trained
        model.fit(X = self.X, y = self.y)
        # run prediction
        pred = model.predict(self.X)
        # get MSE
        score = self.mse(pred, self.y)
        self.assertTrue(score < 0.1)
    '''

if __name__ == "__main__":
    unittest.main()
