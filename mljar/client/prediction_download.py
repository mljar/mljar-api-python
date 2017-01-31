import os
import uuid
import pandas as pd
from base import MljarHttpClient
from ..exceptions import PredictionDownloadException

from ..log import logger

class PredictionDownloadClient(MljarHttpClient):
    '''
    Client to get predictions from MLJAR.
    '''
    def __init__(self):
        self.url = "/download/prediction/"
        super(PredictionDownloadClient, self).__init__()

    def download(self, prediction_hid):
        response = self.request("POST", self.url, data = {"prediction_id": prediction_hid}, parse_json=False)
        pred = None
        try:
            tmp_file = '/tmp/mljar_prediction_' + str(uuid.uuid4()) + '.csv'
            with open(tmp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
            pred = pd.read_csv(tmp_file)
            os.remove(tmp_file)
        except Exception as e:
            raise PredictionDownloadException(str(e))
        return pred
