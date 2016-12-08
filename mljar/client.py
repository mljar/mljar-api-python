import os
import json, requests

# apt-get install libffi-dev libssl-dev
# pip install requests[security]
# pip install pyopenssl ndg-httpsclient pyasn1

API_VERSION = 'v1'
MLJAR_ENDPOINT = 'http://0.0.0.0:3000/api'

class Error(Exception):
    """Base exception class for this module"""
    pass

class TokenError(Error):
    pass

class DataReadError(Error):
    pass

class JSONReadError(Error):
    pass

class NotFoundError(Error):
    pass


class Client(object):
    '''
        API Client
    '''

    def __init__(self, token = None, api_endpoint = None):
        if not token:
            self.TOKEN = os.environ.get('MLJAR_TOKEN', None)
        else:
            self.TOKEN = token
        if not self.TOKEN:
            raise TokenError()

        if not api_endpoint:
            self.API_ENDPOINT = os.environ.get('MLJAR_API_ENDPOINT', None)
        else:
            self.API_ENDPOINT = api_endpoint

        if not self.API_ENDPOINT:
            self.API_ENDPOINT = MLJAR_ENDPOINT

        self._urls = {
            'project':   '/'.join([self.API_ENDPOINT, API_VERSION, 'projects']),
            's3policy':  '/'.join([self.API_ENDPOINT, API_VERSION, 's3policy/']),

        }
        print self._urls, self.TOKEN

    def _make_request(self, url_name = '', custom_url = '', request_type = 'get', url_additional = '', input_json = {}, with_header = True):
        try:
            response = None

            query_parameters = {}
            headers = {'Authorization': 'Token '+self.TOKEN}
            my_url = ''
            if url_name in self._urls:
                my_url = self._urls[url_name]
            if custom_url != '':
                my_url = custom_url
            if my_url == '':
                raise Exception('Wrong URL address')

            if request_type == 'get':
                if url_additional != '':
                    my_url += '/' + url_additional
                response = requests.get(my_url, headers=headers)
                print 'url', response.url
            elif request_type == 'post':
                if with_header:
                    response = requests.post(my_url, data=input_json, headers=headers)
                else:
                    response = requests.post(my_url, data=input_json)
        except Exception as e:
            print 'There was an error during API call, %s' % str(e)
        finally:
            return response

    def _get_data(self, response):
        if response is None:
            return None
        if response.status_code == 404:
            raise NotFoundError()

        try:
            data = response.json()
        except ValueError as e:
            raise JSONReadError('Get data failed, %s' % str(e) )

        if not response.ok:
            msg = [data[m] for m in ("id", "message") if m in data][1]
            raise DataReadError(msg)

        return data

class MljarClient(Client):
    '''
        Mljar API Client
    '''
    def get_projects(self):
        response = self._make_request(url_name = 'project', request_type = 'get')
        projects = self._get_data(response)

        if projects is None:
            print 'There are some problems with projects fetch'
            return

        print 'There are', len(projects), 'projects'
        for proj in projects:
            print 'Project title:', proj['title'], 'hid:', proj['hid']

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
        print 'Get project details', project_hid
        response = self._make_request(url_name = 'project', request_type = 'get', url_additional = project_hid)
        print 'Response', response
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

        print 'Create project with data', data

        response = self._make_request(url_name = 'project', request_type = 'post', input_json = data)
        print 'Response', response
        if response.status_code == 201:
            print 'Project successfully created'
        details = self._get_data(response)
        return details

    def _get_s3_policy(self, project_hid):
        print 'Get S3 Policy', project_hid
        response = self._make_request(url_name = 's3policy', request_type = 'post', input_json = {'project_hid':project_hid})
        print 'Response', response
        details = self._get_data(response)
        print 'Details', details, type(details)
        return details

    def _upload_file_to_s3(self, storage_url, access_key, file_policy, signature, dst_dataset_dir, file_path):
        print 'Upload to S3'
        data = {
            'key': dst_dataset_dir + file_path,
            'AWSAccessKeyId': access_key,
            'acl': 'private',
            'policy': file_policy,
            'signature': signature,
            'file': file_path
        }
        print 'Data', data
        response = self._make_request(custom_url=storage_url, request_type = 'post', input_json = data, with_header=False)
        print 'Response', response
        #details = self._get_data(response)
        #print 'Details', details, type(details)

    def add_new_dataset(self, project_hid, title, file_path, prediction_only = False):
        print 'Add new dataset', project_hid, title, file_path
        policy = self._get_s3_policy(project_hid)
        print 'Policy', policy
        access_key = policy['access_key'];
        file_policy = policy['train_policy'];
        signature = policy['train_signature'];
        dst_dataset_dir = policy['dst_dataset_dir'];
        storage_url = policy['storage_url'];
        print 'Upload file to S3'

        self._upload_file_to_s3(storage_url, access_key, file_policy, signature, dst_dataset_dir, file_path)
