from .base import MljarHttpClient
from ..model.prediction import Prediction
from ..exceptions import NotFoundException

class PredictionClient(MljarHttpClient):
    '''
    Client to interact with MLJAR results (models).
    '''
    def __init__(self, project_hid):
        self.url = "/predictions"
        self.project_hid = project_hid
        super(PredictionClient, self).__init__()

    def get_prediction(self, dataset_hid, result_hid):
        '''
        Get prediction.
        '''
        response = self.request("GET", self.url + '?project_id=' + self.project_hid + '&dataset_id='+dataset_hid+'&result_id='+result_hid)
        predictions_dict = response.json()
        if len(predictions_dict) == 1:
            return Prediction.from_dict(predictions_dict[0])
        return None
