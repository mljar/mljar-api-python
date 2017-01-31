'''
DatasetClient tests.
'''
import os
import unittest
import pandas as pd

from mljar.client.project import ProjectClient
from mljar.client.dataset import DatasetClient

from project_based_test import ProjectBasedTest

class DatasetClientTest(ProjectBasedTest):

    '''
    @staticmethod
    def clean_projects():
        project_client = ProjectClient()
        projects = project_client.get_projects()
        for proj in projects:
            if proj.title.startswith('Test'):
                project_client.delete_project(proj.hid)

    @classmethod
    def setUpClass(cls):
        DatasetClientTest.clean_projects()

    @classmethod
    def tearDownClass(cls):
        DatasetClientTest.clean_projects()
    '''
    def setUp(self):
        proj_title = 'Test project-01'
        proj_task = 'bin_class'
        # setup project
        self.project_client = ProjectClient()
        self.project = self.project_client.create_project(title = proj_title, task = proj_task)
        # load data
        df = pd.read_csv('tests/data/test_1.csv')
        cols = ['sepal length', 'sepal width', 'petal length', 'petal width']
        target = 'class'
        self.X = df[cols]
        self.y = df[target]

    def tearDown(self):
        # clean
        self.project_client.delete_project(self.project.hid)

    def test_get_datasests(self):
        """
        Get empty list of datasets in project.
        """
        # get datasets
        datasets = DatasetClient(self.project.hid).get_datasets()
        self.assertEqual(datasets, [])


    def test_add_dataset_for_training(self):
        # setup dataset client
        dc = DatasetClient(self.project.hid)
        self.assertNotEqual(dc, None)
        # get datasets, there should be none
        datasets = dc.get_datasets()
        self.assertEqual(len(datasets), 0)
        # add dataset
        my_dataset = dc.add_dataset_if_not_exists(self.X, self.y)
        self.assertNotEqual(my_dataset, None)
        # get datasets
        datasets = dc.get_datasets()
        self.assertEqual(len(datasets), 1)
        my_dataset_2 = dc.get_dataset(my_dataset.hid)
        self.assertEqual(my_dataset.hid, my_dataset_2.hid)
        self.assertEqual(my_dataset.title, my_dataset_2.title)

    def test_add_dataset_for_prediction(self):
        # setup dataset client
        dc = DatasetClient(self.project.hid)
        self.assertNotEqual(dc, None)
        # get datasets, there should be none
        datasets = dc.get_datasets()
        self.assertEqual(len(datasets), 0)
        # add dataset
        my_dataset = dc.add_dataset_if_not_exists(self.X, None)
        self.assertNotEqual(my_dataset, None)
        # get datasets
        datasets = dc.get_datasets()
        self.assertEqual(len(datasets), 1)
        my_dataset_2 = dc.get_dataset(my_dataset.hid)
        self.assertEqual(my_dataset.hid, my_dataset_2.hid)
        self.assertEqual(my_dataset.title, my_dataset_2.title)


    def test_add_existing_dataset(self):
        # setup dataset client
        dc = DatasetClient(self.project.hid)
        self.assertNotEqual(dc, None)
        # get initial number of datasets
        init_datasets_cnt = len(dc.get_datasets())
        # add dataset
        dc.add_dataset_if_not_exists(self.X, self.y)
        # get datasets
        datasets = dc.get_datasets()
        self.assertEqual(len(datasets), init_datasets_cnt+1)
        # add the same dataset
        # it shouldn't be added
        dc.add_dataset_if_not_exists(self.X, self.y)
        # number of all datasets in project should be 1
        datasets = dc.get_datasets()
        self.assertEqual(len(datasets), init_datasets_cnt+1)
