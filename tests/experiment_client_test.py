'''
ExperimentClient tests.
'''
import os
import unittest
import pandas as pd

from mljar.client.project import ProjectClient
from mljar.client.dataset import DatasetClient
from mljar.client.experiment import ExperimentClient

from project_based_test import ProjectBasedTest

class ExperimentClientTest(ProjectBasedTest):

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
        dc = DatasetClient(self.project.hid)
        # add dataset
        self.dataset = dc.add_dataset_if_not_exists(df[cols], df[target])

    def tearDown(self):
        # clean
        self.project_client.delete_project(self.project.hid)

    def test_create(self):
        """
        Create experiment test.
        """
        # add experiment
        ec = ExperimentClient(self.project.hid)
        self.assertNotEqual(ec, None)
        experiment = ec.create_experiment(self.dataset, self.expt_title, self.project.task,
                                            self.validation, self.algorithms, self.metric,
                                            self.tuning_mode, self.time_constraint, self.create_enseble)
        self.assertNotEqual(experiment, None)

'''
(self, train_dataset, experiment_title, project_task, \
                                    validation, algorithms, metric, \
                                    tuning_mode, time_constraint, create_enseble):

get_experiments

get_experiment

create

create_experiment_if_not_exists

'''
