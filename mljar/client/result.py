from base import MljarHttpClient
from ..model.result import Result
from ..exceptions import NotFoundException

class ResultClient(MljarHttpClient):
    '''
    Client to interact with MLJAR results (models).
    '''
    def __init__(self, project_hid):
        self.url = "/results/"
        self.project_hid = project_hid
        super(ResultClient, self).__init__()

    def get_results(self, experiment_hid = None):
        '''
        List all models.
        '''
        data = {'project_id': self.project_hid}
        if experiment_hid is not None:
            data['experiment_id'] = experiment_hid
        response = self.request("POST", self.url, data = data)
        results_dict = response.json()
        return [Result.from_dict(r) for r in results_dict]
