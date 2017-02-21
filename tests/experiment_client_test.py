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
        # there should be none experiments
        experiments = ec.get_experiments()
        self.assertEqual(experiments, [])
        # create new experiment
        experiment = ec.add_experiment_if_not_exists(self.dataset, self.expt_title, self.project.task,
                                            self.validation, self.algorithms, self.metric,
                                            self.tuning_mode, self.time_constraint, self.create_enseble)
        self.assertNotEqual(experiment, None)
        self.assertEqual(experiment.title, self.expt_title)
        self.assertEqual(experiment.validation_scheme, self.validation)
        self.assertEqual(experiment.metric, self.metric)
        # get all experiments, should be only one
        experiments = ec.get_experiments()
        self.assertEqual(len(experiments), 1)
        # get experiment by hid, there should be the same
        experiment_2 = ec.get_experiment(experiment.hid)
        self.assertEqual(experiment_2.hid, experiment.hid)
        self.assertEqual(experiment_2.title, experiment.title)
        self.assertEqual(experiment_2.metric, experiment.metric)
        self.assertEqual(experiment_2.validation_scheme, experiment.validation_scheme)
        self.assertTrue(experiment.equal(experiment_2))
        # test __str__ method
        self.assertTrue('id' in str(experiment_2))
        self.assertTrue('title' in str(experiment_2))
        self.assertTrue('metric' in str(experiment_2))
        self.assertTrue('validation' in str(experiment_2))

    def test_create_if_exists(self):
        """
        Create experiment after experiment is already in project.
        """
        # add experiment
        ec = ExperimentClient(self.project.hid)
        self.assertNotEqual(ec, None)
        # there should be none experiments
        experiments = ec.get_experiments()
        self.assertEqual(experiments, [])
        # create new experiment
        experiment = ec.add_experiment_if_not_exists(self.dataset, self.expt_title, self.project.task,
                                            self.validation, self.algorithms, self.metric,
                                            self.tuning_mode, self.time_constraint, self.create_enseble)
        self.assertNotEqual(experiment, None)
        # get all experiments, should be only one
        experiments = ec.get_experiments()
        self.assertEqual(len(experiments), 1)
        # try to create the same experiment
        experiment_2 = ec.add_experiment_if_not_exists(self.dataset, self.expt_title, self.project.task,
                                            self.validation, self.algorithms, self.metric,
                                            self.tuning_mode, self.time_constraint, self.create_enseble)
        self.assertNotEqual(experiment, None)
        # get all experiments, should be only one
        experiments = ec.get_experiments()
        self.assertEqual(len(experiments), 1)
        # both should be the same
        self.assertEqual(experiment_2.hid, experiment.hid)
        self.assertEqual(experiment_2.title, experiment.title)
        self.assertEqual(experiment_2.metric, experiment.metric)
        self.assertEqual(experiment_2.validation_scheme, experiment.validation_scheme)
        self.assertTrue(experiment.equal(experiment_2))
