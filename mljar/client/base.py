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
            except Exception as e:
                message = None
            logger.debug("Error received : status_code: {}, message: {}".format(response.status_code,
                                                                                      message or response.content))

            if response.status_code == 401:
                raise AuthenticationException()
            elif response.status_code == 404:
                raise NotFoundException()
            elif response.status_code == 400:
                raise BadRequestException(response.content)
            elif response.status_code == 500:
                raise MljarException('server error: ' +str(response.content))
            else:
                response.raise_for_status()
