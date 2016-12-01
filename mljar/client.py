import os
import json, requests

# apt-get install libffi-dev libssl-dev
# pip install requests[security]
# pip install pyopenssl ndg-httpsclient pyasn1

API_VERSION = 'v1'

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
            self.API_ENDPOINT = 'https://mljar.com'

        self._urls = {
            'project': '/'.join([self.API_ENDPOINT, API_VERSION, 'projects'])
        }
        #print self._urls, self.TOKEN

    def _make_request(self, url_name, request_type = 'get'):
        try:
            response = None
            input_json = {}
            query_parameters = {}
            headers = {'Authorization': 'Token '+self.TOKEN}
            if request_type == 'get':
                response = requests.get(self._urls[url_name], headers=headers)
            elif request_type == 'post':
                response = requests.post(self._urls[url_name], data=input_json, headers=headers, params=query_parameters)
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
            print 'Project:', proj['title']
