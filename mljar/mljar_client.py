
import os
import json, requests
from client import Client

class MljarClient(Client):
    '''
        Mljar API Client
    '''
    def __init__(self):
        super(MljarClient, self).__init__()

    def get_projects(self, verbose = True):
        response = self._make_request(url_name = 'project', request_type = 'get')
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
            return 'Unknown task'
        return tasks[task_short]

    def get_project_details(self, project_hid, verbose = True):
        '''
            Print out project details and return details json.
        '''
        response = self._make_request(url_name = 'project', request_type = 'get', url_additional = '/' + project_hid)
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
        data={'hardware': 'cloud',
                'scope': 'private',
                'task': task,
                'compute_now': 0,
                'description': description,
                'title':title}
        response = self._make_request(url_name = 'project', request_type = 'post', input_json = data)
        if response.status_code == 201:
            print 'Project successfully created'
        details = self._get_data(response)
        return details

    def get_datasets(self, project_hid):
        response = self._make_request(url_name = 'dataset', request_type = 'get', url_additional = '?project_id='+project_hid)
        details = self._get_data(response)
        return details

    def accept_dataset_column_usage(self, dataset_hid):
        data = {'dataset_id': dataset_hid}
        response = self._make_request(url_name = 'accept_column_usage', request_type = 'post', input_json = data)
        details = self._get_data(response)
        return details


    def _get_signed_url(self, project_hid, file_path):
        response = self._make_request(url_name = 's3policy', request_type = 'post',
                                        input_json = {'project_hid':project_hid,
                                                        'fname': file_path.split('/')[-1]})
        details = self._get_data(response)
        return details

    def _upload_file_to_s3(self, signed_url, file_path):
        response = requests.put(signed_url, data=open(file_path, 'rb').read())
        if response.status_code == 200:
            print 'Uploaded successfully'
        response.raise_for_status()

    def add_new_dataset(self, project_hid, title, file_path, prediction_only=False):
        print 'Add new dataset:', project_hid, title, file_path
        url_data = self._get_signed_url(project_hid, file_path)

        signed_url = url_data['signed_url']
        dst_path   = url_data['destination_path']
        self._upload_file_to_s3(signed_url, file_path)

        print 'Uploaded to', dst_path

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
            'prediction_only': 0
        }
        response = self._make_request(url_name = 'dataset', request_type = 'post', input_json = data)
        print 'Response', response
        if response.status_code == 201:
            print 'Dataset successfully created'
        details = self._get_data(response)
        return details

    def create_experiment(self, data):
        response = self._make_request(url_name = 'experiment', request_type = 'post', input_json = data)
        if response.status_code == 201:
            print 'Experiment successfully created'
        details = self._get_data(response)
        return details
