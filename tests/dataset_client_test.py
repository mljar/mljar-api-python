'''
DatasetClient tests.
'''
import os
import unittest
import pandas as pd
import numpy as np

from mljar.client.project import ProjectClient
from mljar.client.dataset import DatasetClient

from .project_based_test import ProjectBasedTest

class DatasetClientTest(ProjectBasedTest):

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
        self.X = df.loc[:,cols]
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

    def test_prepare_data(self):
        """ Test _prepare_data method on numpy array data """
        dc = DatasetClient(self.project.hid)
        samples = 100
        columns = 10
        X = np.random.rand(samples, columns)
        y = np.random.choice([0,1], samples, replace = True)
        data, data_hash = dc._prepare_data(X, y)
        self.assertTrue(data is not None)
        self.assertTrue(data_hash is not None)
        self.assertTrue(isinstance(data_hash, str))
        self.assertEqual(11, len(data.columns))
        self.assertTrue('target' in data.columns)
        self.assertTrue('attribute_1' in data.columns)
        self.assertTrue('attribute_10' in data.columns)

    def test_get_dataset_for_wrong_hid(self):
        """ Get dataset for wrong hid should return None """
        dc = DatasetClient(self.project.hid)
        dataset = dc.get_dataset('some-wrong-hid')
        self.assertTrue(dataset is None)

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
        # test __str__ method
        self.assertTrue('id' in str(my_dataset_2))
        self.assertTrue('title' in str(my_dataset_2))
        self.assertTrue('file' in str(my_dataset_2))


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


    def test_prepare_data_two_sources(self):
        dc = DatasetClient(self.project.hid)
        data_1, data_hash_1 = dc._prepare_data(self.X, self.y)
        data_2, data_hash_2 = dc._prepare_data(self.X, None)
        self.assertNotEqual(data_hash_1, data_hash_2)


    def test_prepare_data_two_sources_numpy(self):
        dc = DatasetClient(self.project.hid)
        data_1, data_hash_1 = dc._prepare_data(np.array(self.X), np.array(self.y))
        data_2, data_hash_2 = dc._prepare_data(np.array(self.X), None)
        self.assertNotEqual(data_hash_1, data_hash_2)

    def test_create_and_delete(self):
        # setup dataset client
        dc = DatasetClient(self.project.hid)
        self.assertNotEqual(dc, None)
        # get initial number of datasets
        init_datasets_cnt = len(dc.get_datasets())
        # add dataset
        my_dataset_1 = dc.add_dataset_if_not_exists(self.X, self.y)
        my_dataset_2 = dc.add_dataset_if_not_exists(self.X, y = None)
        # get datasets
        datasets = dc.get_datasets()
        self.assertEqual(len(datasets), init_datasets_cnt+2)
        # delete added dataset
        dc.delete_dataset(my_dataset_1.hid)
        # check number of datasets
        datasets = dc.get_datasets()
        self.assertEqual(len(datasets), init_datasets_cnt+1)


if __name__ == "__main__":
    unittest.main()
