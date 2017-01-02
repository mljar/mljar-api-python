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
            raise TokenError('Please define environment variable MLJAR_TOKEN')

        if not api_endpoint:
            self.API_ENDPOINT = os.environ.get('MLJAR_API_ENDPOINT', None)
        else:
            self.API_ENDPOINT = api_endpoint

        if not self.API_ENDPOINT:
            self.API_ENDPOINT = MLJAR_ENDPOINT

        self._urls = {
            'project':      '/'.join([self.API_ENDPOINT, API_VERSION, 'projects']),
            'dataset':      '/'.join([self.API_ENDPOINT, API_VERSION, 'datasets']),
            'experiment':   '/'.join([self.API_ENDPOINT, API_VERSION, 'experiments']),
            'result':       '/'.join([self.API_ENDPOINT, API_VERSION, 'results/']),
            'predict':       '/'.join([self.API_ENDPOINT, API_VERSION, 'predict/']),
            's3policy':     '/'.join([self.API_ENDPOINT, API_VERSION, 's3policy/']),
            'accept_column_usage': '/'.join([self.API_ENDPOINT, API_VERSION, 'accept_column_usage/']),
        }


    def _make_request(self, url_name = '', custom_url = '', request_type = 'get', url_additional = '', input_json = {}, with_header = True):
        try:
            response = None
            headers = {'Authorization': 'Token '+self.TOKEN } #'Content-Type': 'application/json'
            my_url = ''
            if url_name in self._urls:
                my_url = self._urls[url_name]
            if custom_url != '':
                my_url = custom_url
            if my_url == '':
                raise Exception('Wrong URL address')
            if url_additional != '':
                my_url += url_additional

            if request_type == 'get':
                response = requests.get(my_url, headers=headers)
            elif request_type == 'post':
                if with_header:
                    response = requests.post(my_url, data=input_json, headers=headers)
                else:
                    response = requests.post(my_url, data=input_json)
            elif request_type == 'put':
                if with_header:
                    response = requests.put(my_url, data=input_json, headers=headers)
                else:
                    response = requests.put(my_url, data=input_json)

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
