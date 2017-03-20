import numpy as np
import pandas as pd
import uuid
import os
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED
from os.path import basename
from base import MljarHttpClient
from ..model.dataset import Dataset
from ..exceptions import NotFoundException, MljarException, CreateDatasetException, DatasetUnknownException

from dataupload import DataUploadClient
from ..log import logger

from ..utils import make_hash

class DatasetClient(MljarHttpClient):
    '''
    Client to interact with MLJAR datasets
    '''
    def __init__(self, project_hid):
        self.project_hid = project_hid
        self.url = "/datasets"
        super(DatasetClient, self).__init__()

    def get_datasets(self):
        '''
        Gets all datasets in the project
        '''
        logger.info('Get datasets, project id {}'.format(self.project_hid))
        response = self.request("GET", self.url+'?project_id='+self.project_hid)
        datasets_dict = response.json()
        return [Dataset.from_dict(ds) for ds in datasets_dict]

    def get_dataset(self, dataset_hid):
        '''
        Gets dataset for specified hid
        '''
        logger.info('Get dataset, dataset id {}'.format(dataset_hid))
        try:
            response = self.request("GET", self.url+'/'+dataset_hid)
            return Dataset.from_dict(response.json())
        except NotFoundException:
            logger.error('Dataset not found')
            return None


    def _prepare_data(self, X, y):
        '''
        Concatenates matrices and computes hash
        '''
        logger.info('Prepare dataset and compute hash')
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
                data = X
                data['target'] = y
                # todo: add search for target like attributes and rename
                # "target", "class", "loss"
            else:
                data = X

        dataset_hash = str(make_hash(data))
        return data, dataset_hash

    def _wait_till_all_datasets_are_valid(self):
        '''
        Waits till all datasets is valid. If all valid it returns True,
        if wait time is exceeded and there is any dataset not valid then
        it returns False.
        '''
        logger.info('Wait till all datasets are valid')
        total_checks = 120
        for i in xrange(total_checks):
            datasets = self.get_datasets()
            if datasets is not None:
                logger.info('There are %s datasets' % len(datasets))
                if len(datasets) == 0:
                    logger.info('No datasets to wait for')
                    return
                not_validated = [ds for ds in datasets if ds.valid == 0]
                if len(not_validated) == 0:
                    logger.info('All datasets are valid')
                    return
            else:
                logger.info('None datasets list')
            time.sleep(5)
        raise MljarException('There are some problems with reading one of your dataset. \
                            Please login to mljar.com and check your project for more details.')



    def add_dataset_if_not_exists(self, X, y, title_prefix = 'dataset-'):
        '''
        Checks if dataset already exists, if not it add dataset to project.
        '''
        logger.info('Add dataset if not exists')
        # before start adding any new dataset
        # wait till all dataset are validated
        # it does not return an object, it just waits
        self._wait_till_all_datasets_are_valid()
        logger.info('All datasets are valid till now')
        # check if dataset already exists
        data, dataset_hash = self._prepare_data(X, y)
        datasets = self.get_datasets()
        dataset_details = [d for d in datasets if d.dataset_hash == dataset_hash]
        # dataset with specified hash does not exist
        if len(dataset_details) != 1:
            # add new dataset
            dataset_details = self.add_new_dataset(data, y, title_prefix)
        else:
            dataset_details = dataset_details[0]

        if dataset_details is None:
            raise MljarException('There was a problem during new dataset addition')
        # wait till dataset is validated ...
        self._wait_till_all_datasets_are_valid()
        if not self._accept_dataset_column_usage(dataset_details.hid):
            raise MljarException('There was a problem with accept column usage for your dataset.')
        # get dataset with updated statistics
        my_dataset = self.get_dataset(dataset_details.hid)
        if my_dataset is None:
            raise DatasetUnknownException('Can not find dataset: %s' % self.dataset_title)
        if my_dataset.valid != 1:
            raise MljarException('Sorry, your dataset can not be read by MLJAR. \
                                    Please report this to us - we will fix it.')

        if my_dataset.column_usage_min is None:
            raise MljarException('Something bad happend! There is no attributes \
                                    usage defined for your dataset')

        return my_dataset

    def _accept_dataset_column_usage(self, dataset_hid):
        logger.info('Accept column usage')
        response = self.request("POST", '/accept_column_usage/',data = {'dataset_id': dataset_hid})
        return response.status_code == 200


    def add_new_dataset(self, data, y, title_prefix = 'dataset-'):
        logger.info('Add new dataset')
        title = title_prefix + str(uuid.uuid4())[:4] # set some random name
        file_path = '/tmp/dataset-'+ str(uuid.uuid4())[:8]+'.csv'

        logger.info('Compress data before export')
        prediction_only = y is None
        # save to local storage
        data.to_csv(file_path, index=False)
        # compress
        file_path_zip = file_path + '.zip'
        with ZipFile(file_path_zip, 'w', ZIP_DEFLATED) as myzip:
            myzip.write(file_path, basename(file_path))

        # upload data to MLJAR storage
        dst_path = DataUploadClient().upload_file(self.project_hid, file_path_zip)
        # create a dataset instance in DB
        data = {
            'title': title,
            'file_path': dst_path,
            'file_name': basename(file_path_zip),
            'file_size': round(os.path.getsize(file_path_zip) / 1024.0/ 1024.0, 2),
            'derived': 0,
            'valid': 0,
            'parent_project': self.project_hid,
            'meta': '',
            'data_type': 'tabular',
            'scope': 'private',
            'prediction_only': 1 if prediction_only else 0
        }
        logger.info('Add information about dataset into MLJAR')
        response = self.request("POST", self.url, data = data)
        if response.status_code != 201:
            raise CreateDatasetException()
        logger.info('Clean tmp files')
        # clean data file
        os.remove(file_path)
        os.remove(file_path_zip)

        return Dataset.from_dict(response.json())
