import os
import json, requests

from .. import API_VERSION, MLJAR_ENDPOINT
from ..exceptions import MljarException, TokenException, DataReadException, BadRequestException
from ..exceptions import JSONReadException, NotFoundException, AuthenticationException


from ..log import logger

class MljarHttpClient(object):
    '''
        Mljar Client for HTTP Requests.
    '''

    def __init__(self):
        self.TOKEN = os.environ.get('MLJAR_TOKEN', None)
        if not self.TOKEN:
            raise TokenException('Please define environment variable MLJAR_TOKEN. \
                                You can get you MLJAR token by login to mljar.com account. \
                                It is available in your settings.')

        self.base_url = '/'.join([MLJAR_ENDPOINT, API_VERSION])

    def request(self, method, url, data=None, with_header=True, url_outside_mljar=False, parse_json=True):
        """
        Execute the request using requests library.
        """
        if url_outside_mljar:
            request_url = url
        else:
            request_url = self.base_url + url
        logger.debug("Starting request to url: {} with data: {}".format(request_url, data))

        headers = {'Authorization': 'Token '+self.TOKEN }
        if with_header:
            response = requests.request(method, request_url, headers=headers, data=data)
        else:
            response = requests.request(method, request_url, data=data)

        if parse_json:
            try:
                if response.status_code != 204:
                    logger.debug("Response content: {}, headers: {}".format(response.json(), response.headers))
            except Exception as e:
                logger.error("Request failed: {} {}".format(response.content, str(e)))
        self._check_response_status(response)
        return response

    def _check_response_status(self, response):
        """
        Check if response is successful else raise Exception.
        """
        if not (200 <= response.status_code < 300):
            try:
                message = response.json()["errors"]
            except Exception:
                message = None
            logger.debug("Error received : status_code: {}, message: {}".format(response.status_code,
                                                                                      message or response.content))

            if response.status_code == 401:
                raise AuthenticationException()
            elif response.status_code == 404:
                raise NotFoundException()
            elif response.status_code == 400:
                raise BadRequestException()
            else:
                response.raise_for_status()


'''
def _get_data(self, response):
    if response is None:
        return None

    try:
        data = response.json()
    except ValueError as e:
        raise JSONReadException('Get data failed, %s' % str(e) )

    if not response.ok:
        msg = [data[m] for m in ("id", "message") if m in data][1]
        raise DataReadException(msg)

    return data
'''

'''

        self._urls = {
            'project':      '/'.join([self.API_ENDPOINT, API_VERSION, 'projects']),
            'dataset':      '/'.join([self.API_ENDPOINT, API_VERSION, 'datasets']),
            'experiment':   '/'.join([self.API_ENDPOINT, API_VERSION, 'experiments']),
            'result':       '/'.join([self.API_ENDPOINT, API_VERSION, 'results/']),
            'predict':      '/'.join([self.API_ENDPOINT, API_VERSION, 'predict/']),
            'predictions':  '/'.join([self.API_ENDPOINT, API_VERSION, 'predictions']), # it is not a bug, we don't need here '/'
            'download_prediction': '/'.join([self.API_ENDPOINT, API_VERSION, 'download/prediction/']),
            's3policy':     '/'.join([self.API_ENDPOINT, API_VERSION, 's3policy/']),
            'accept_column_usage': '/'.join([self.API_ENDPOINT, API_VERSION, 'accept_column_usage/']),
        }


    def _make_request(self, url_name = '', custom_url = '', request_type = 'get', url_additional = '', input_json = {}, with_header = True):
        try:
            response = None
            headers = {'Authorization': 'Token '+self.TOKEN } #'Content-Type': 'application/json'
            my_url = ''
            if url_name in self._urls:
            if my_url == '':
                raise Exception('Wrong URL address')
            if url_additional != '':
                my_url += url_additional

            #print 'request', my_url, request_type, input_json, with_header, headers

            if request_type == 'get':
                response = requests.get(my_url, headers=headers)
            elif request_type == 'post':
                if with_header:
                    response = requests.post(my_url, data=input_json, headers=headers)
                else:
                    response = requests.post(my_url, data=input_json)
            elif request_type == 'put':
                if with_header:
                    print my_url, 'with header', input_json
                    response = requests.put(my_url, data=input_json, headers=headers)
                else:
                    response = requests.put(my_url, data=input_json)

        except MljarException as e:
            print 'There was an error during API call, %s' % str(e)
        finally:
            return response
'''
