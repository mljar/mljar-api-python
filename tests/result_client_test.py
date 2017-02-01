'''
ResultClient tests.
'''
import os
import unittest
import pandas as pd
import time

from mljar.client.project import ProjectClient
from mljar.client.dataset import DatasetClient
from mljar.client.experiment import ExperimentClient
from mljar.client.result import ResultClient

from project_based_test import ProjectBasedTest

class ResultClientTest(ProjectBasedTest):

    def setUp(self):
        proj_title = 'Test project-01'
        proj_task = 'bin_class'
        self.expt_title = 'Test experiment-01'
        self.validation = '5fold'
        self.algorithms = ['xgb']
        self.metric = 'logloss'
        self.tuning_mode = 'Normal'
        self.time_constraint = 1
        self.create_enseble = False
        # setup project
        self.project_client = ProjectClient()
        self.project = self.project_client.create_project(title = proj_title, task = proj_task)
        # load data
        df = pd.read_csv('tests/data/test_1.csv')
        cols = ['sepal length', 'sepal width', 'petal length', 'petal width']
        target = 'class'
        # add dataset
        self.dataset = DatasetClient(self.project.hid).add_dataset_if_not_exists(df[cols], df[target])


    def tearDown(self):
        # clean
        self.project_client.delete_project(self.project.hid)

    def test_get_results_for_wrong_project(self):
        # init result client
        rc = ResultClient('wrong-hid')
        self.assertNotEqual(rc, None)
        # get results - should be empty
        results = rc.get_results()
        self.assertEqual(results, [])

    def test_get_results_for_project(self):
        # init result client
        rc = ResultClient(self.project.hid)
        self.assertNotEqual(rc, None)
        # get results - should be empty
        results = rc.get_results()
        self.assertEqual(results, [])
        # add experiment
        ec = ExperimentClient(self.project.hid)
        # create new experiment
        self.experiment = ec.add_experiment_if_not_exists(self.dataset, self.expt_title, self.project.task,
                                            self.validation, self.algorithms, self.metric,
                                            self.tuning_mode, self.time_constraint, self.create_enseble)
        # wait some time till models are initialized
        time.sleep(120)
        # get results - should be some models there
        results = rc.get_results()
        self.assertNotEqual(len(results), 0)


    def test_get_results_for_experiment(self):
        # init result client
        rc = ResultClient(self.project.hid)
        self.assertNotEqual(rc, None)
        # get results - should be empty
        results = rc.get_results()
        self.assertEqual(results, [])
        # get results for wrong experiment hid
        results = rc.get_results('wrong-hid')
        self.assertEqual(results, [])
        # add experiment
        ec = ExperimentClient(self.project.hid)
        # create new experiment
        self.experiment = ec.add_experiment_if_not_exists(self.dataset, self.expt_title, self.project.task,
                                            self.validation, self.algorithms, self.metric,
                                            self.tuning_mode, self.time_constraint, self.create_enseble)
        # wait some time till models are initialized
        time.sleep(120)
        # get results for experiment - should be some models there
        results = rc.get_results(self.experiment.hid)
        self.assertNotEqual(len(results), 0)

        # get results for project
        project_results = rc.get_results()
        self.assertNotEqual(results, [])
        # get results for wrong experiment hid
        # all results from project should be returned
        results_2 = rc.get_results('wrong-hid')
        self.assertEqual(len(project_results), len(results_2))
