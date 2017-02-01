import json
from base import MljarHttpClient
from ..exceptions import FileUploadException

from ..log import logger

class PredictJobClient(MljarHttpClient):
    '''
    Client to submit predict job in MLJAR.
    '''
    def __init__(self):
        self.url = "/predict/"
        super(PredictJobClient, self).__init__()


    def submit(self, project_hid, dataset_hid, result_hid):
        data =  {
                    'predict_params' : json.dumps({'project_id': project_hid,
                                                    'project_hardware': 'cloud',
                                                    'algorithms_ids': [result_hid],
                                                    'dataset_id': dataset_hid,
                                                    'cv_models':1})
                }
        response = self.request("POST", self.url, data = data, parse_json = False)
        return response.status_code == 200
