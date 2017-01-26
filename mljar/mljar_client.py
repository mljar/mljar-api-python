import pandas as pd
import os
import json, requests
from client import Client
import uuid


'''
    MljarClient errors:
'''
class Error(Exception):
    pass

class UnknownPorjectTaskError(Error):
    pass

class ProjectCreateError(Error):
    pass

class FileUploadError(Error):
    pass

class DatasetCreateError(Error):
    pass

class ExperimentCreateError(Error):
    pass


class MljarClient(Client):
    '''
        Mljar API Client.
    '''
    def __init__(self):
        super(MljarClient, self).__init__()
        self.verbose = True

    def get_projects(self, verbose = True):
        '''
            List all user projects.
        '''
        response = self._make_request(url_name = 'project', request_type = 'get')
        response.raise_for_status()
        projects = self._get_data(response)
        if projects is None:
            print 'There are some problems with projects fetch'
            return None
        if verbose:
            print 'There are', len(projects), 'projects'
            for proj in projects:
                print 'Project title:', proj['title'], 'hid:', proj['hid']
        return projects

    def _task_to_full_name(self, task_short):
        tasks = {'bin_class': "Binary classification",
                    'reg': "Regression",
                    'img_class': "Images classification"}
        if task_short not in tasks:
            raise UnknownPorjectTaskError('Unknown task %s' % task_short)
        return tasks[task_short]

    def get_project_details(self, project_hid, verbose = False):
        '''
            Print out project details and return details in json.
        '''
        response = self._make_request(url_name = 'project', request_type = 'get', url_additional = '/' + project_hid)
        response.raise_for_status()
        details = self._get_data(response)
        if verbose:
            print '-'*50,'\nProject details\n','-'*50
            print 'Title:', details['title']
            if details['description']:
                print 'Description:', details['description']
            print 'Task:', self._task_to_full_name(details['task'])
            print 'Hardware:', details['hardware']
            print 'User data sources count:', len(details['datasets'])
            print 'Models count:', details['models_cnt']
            print '-'*50
        return details

    def create_project(self, title, description = '', task = 'bin_class'):
        data= {'hardware': 'cloud',
                'scope': 'private',
                'task': task,
                'compute_now': 0,
                'description': description,
                'title':title}

        response = self._make_request(url_name = 'project', request_type = 'post', input_json = data)
        response.raise_for_status()
        if response.status_code == 201:
            print 'Project successfully created.'
        else:
            raise ProjectCreateError('Error during project creation')
        details = self._get_data(response)
        return details

    def get_datasets(self, project_hid):
        response = self._make_request(url_name = 'dataset', request_type = 'get', url_additional = '?project_id='+project_hid)
        response.raise_for_status()
        details = self._get_data(response)
        return details

    def accept_dataset_column_usage(self, dataset_hid):
        data = {'dataset_id': dataset_hid}
        response = self._make_request(url_name = 'accept_column_usage', request_type = 'post', input_json = data)
        response.raise_for_status()
        details = self._get_data(response)
        return details


    def _get_signed_url(self, project_hid, file_path):
        response = self._make_request(url_name = 's3policy', request_type = 'post',
                                        input_json = {'project_hid':project_hid,
                                                        'fname': file_path.split('/')[-1]})
        response.raise_for_status()
        details = self._get_data(response)
        return details

    def _upload_file_to_s3(self, signed_url, file_path):
        response = requests.put(signed_url, data=open(file_path, 'rb').read())
        response.raise_for_status()
        if response.status_code == 200:
            print 'File successfully uploaded'
        else:
            print 'There was a problem with file upload'
            raise FileUploadError('There was a problem with data upload into MLJAR')

    def add_new_dataset(self, project_hid, title, file_path, prediction_only=False):
        # upload data to s3
        url_data = self._get_signed_url(project_hid, file_path)
        signed_url = url_data['signed_url']
        dst_path   = url_data['destination_path']
        self._upload_file_to_s3(signed_url, file_path)
        # create a dataset instance in DB
        data = {
            'title': title,
            'file_path': dst_path,
            'file_name': file_path.split('/')[-1],
            'file_size': round(os.path.getsize(file_path) / 1024.0/ 1024.0, 2),
            'derived': 0,
            'valid': 0,
            'parent_project': project_hid,
            'meta': '',
            'data_type': 'tabular',
            'scope': 'private',
            'prediction_only': 1 if prediction_only else 0
        }
        response = self._make_request(url_name = 'dataset', request_type = 'post', input_json = data)
        response.raise_for_status()
        if response.status_code == 201:
            print 'Dataset successfully created'
        else:
            raise DatasetCreateError('Error during dataset creation')
        details = self._get_data(response)
        return details

    def create_experiment(self, data):
        response = self._make_request(url_name = 'experiment', request_type = 'post', input_json = data)
        response.raise_for_status()
        if response.status_code == 201:
            print 'Experiment successfully created'
        else:
            raise ExperimentCreateError('Error during experiment creation')
        details = self._get_data(response)
        return details

    def get_experiments(self, project_hid):
        response = self._make_request(url_name = 'experiment', request_type = 'get', url_additional = '?project_id='+project_hid)
        response.raise_for_status()
        details = self._get_data(response)
        return details

    def get_experiment_details(self, experiment_hid):
        '''
            Get details of experiment.
        '''
        response = self._make_request(url_name = 'experiment', request_type = 'get', url_additional = '/' + experiment_hid)
        response.raise_for_status()
        details = self._get_data(response)
        return details

    def get_results(self, project_hid):
        data = {'project_id': project_hid} # , 'minify': False
        response = self._make_request(url_name = 'result', request_type = 'post', input_json = data)
        response.raise_for_status()
        details = self._get_data(response)
        return details

    def submit_predict_job(self, project_hid, dataset_hid, result_hid):
        data =  {'predict_params' : json.dumps({'project_id': project_hid,
                    'project_hardware': 'cloud',
                    'algorithms_ids': [result_hid],
                    'dataset_id': dataset_hid,
                    'cv_models':1})
                }
        response = self._make_request(url_name = 'predict', request_type = 'post', input_json = data)
        response.raise_for_status()
        return response.status_code

    def check_if_prediction_available(self, project_hid, dataset_hid, result_hid):
        url_additional = '?project_id=' + project_hid + '&dataset_id='+dataset_hid+'&result_id='+result_hid
        response = self._make_request(url_name = 'predictions', request_type = 'get', url_additional = url_additional)
        response.raise_for_status()
        details = self._get_data(response)
        if len(details) == 1:
            return True, details[0].get('hid', None)
        return False, None


    def download_prediction(self, prediction_hid):
        response = self._make_request(url_name = 'download_prediction', request_type = 'post', input_json = {"prediction_id": prediction_hid})
        response.raise_for_status()
        try:
            tmp_file = '/tmp/mljar_prediction_' + str(uuid.uuid4()) + '.csv'
            pred = None
            with open(tmp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
            pred = pd.read_csv(tmp_file)
            os.remove(tmp_file)
        except Exception as e:
            print '\nThere was unexpected error during geting predictions.', str(e)
        finally:
            return pred
