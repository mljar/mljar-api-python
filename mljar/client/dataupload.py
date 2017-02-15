from base import MljarHttpClient
from ..model.dataset import Dataset
from ..exceptions import FileUploadException

from ..log import logger

class DataUploadClient(MljarHttpClient):
    '''
    Client to upload data into MLJAR.
    '''
    def __init__(self):
        self.url = "/s3policy/"
        super(DataUploadClient, self).__init__()

    def _get_signed_url(self, project_hid, file_path):
        data = {'project_hid':project_hid, 'fname': file_path.split('/')[-1]}
        response = self.request("POST", self.url, data = data)
        return response.json()

    def upload_file(self, project_hid, file_path):
        logger.info('File upload started')
        url_data = self._get_signed_url(project_hid, file_path)
        signed_url = url_data['signed_url']
        dst_path   = url_data['destination_path']
        response = self.request("PUT", signed_url, data=open(file_path, 'rb').read(),
                                            with_header=False, url_outside_mljar=True,
                                            parse_json=False)

        if response.status_code != 200:
            raise FileUploadException('There was a problem with data upload into MLJAR')
        return dst_path
