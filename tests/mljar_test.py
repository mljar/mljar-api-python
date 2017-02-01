'''
Mljar tests.
'''
import os
import pandas as pd
import numpy as np
import unittest
import time

from mljar.client.project import ProjectClient
from project_based_test import ProjectBasedTest
from mljar import Mljar

class MljarTest(ProjectBasedTest):

    def setUp(self):
        self.proj_title = 'Test project-01'
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

    def test_basic_usage(self):
        '''
        Test the most common usage.
        '''
        model = Mljar(project = self.proj_title, experiment = self.expt_title,
                        algorithms = ['xgb'], metric='logloss',
                        validation='3fold', tuning_mode='Normal')
        self.assertNotEqual(model, None)
        # fit models and wait till all models are trained
        model.fit(X = self.X, y = self.y)
        # run prediction
        pred = model.predict(self.X)
        # get MSE
        score = self.mse(pred, self.y)
        self.assertTrue(score < 0.1)

    def test_non_wait_fit(self):
        '''
        Test the non wait fit.
        '''
        model = Mljar(project = self.proj_title, experiment = self.expt_title,
                        algorithms = ['xgb'], metric='logloss',
                        validation='3fold', tuning_mode='Normal')
        self.assertNotEqual(model, None)
        # fit models, just start computation and do not wait
        start_time = time.time()
        model.fit(X = self.X, y = self.y, wait_till_all_done = False)
        end_time = time.time()
        # time to initialize models should not be greater than 5 minutes
        self.assertTrue(end_time - start_time < 5*60)
        # run prediction
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
        self.assertNotEqual(pred, None)
        # get MSE
        score = self.mse(pred, self.y)
        self.assertTrue(score < 0.1)

    def test_retrive_models(self):
        '''
        Test scenario, when user create project, fit models, and try to once
        again run project. In this case, there will be no additional computations,
        all models will be simply retrived from existing project.
        '''
        model = Mljar(project = self.proj_title, experiment = self.expt_title,
                        algorithms = ['xgb'], metric='logloss',
                        validation='3fold', tuning_mode='Normal')
        self.assertNotEqual(model, None)
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
                        algorithms = ['xgb'], metric='logloss',
                        validation='3fold', tuning_mode='Normal')
        self.assertNotEqual(model_2, None)
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
