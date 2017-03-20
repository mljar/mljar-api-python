import json
import warnings
from base import MljarHttpClient
from ..model.experiment import Experiment
from ..exceptions import NotFoundException, MljarException, CreateExperimentException
from ..exceptions import UndefinedExperimentException

from dataupload import DataUploadClient
from ..log import logger

from ..utils import make_hash
from ..utils import MLJAR_METRICS, MLJAR_TUNING_MODES, MLJAR_DEFAULT_ALGORITHMS, MLJAR_DEFAULT_METRICS

class ExperimentClient(MljarHttpClient):
    '''
    Client to interact with MLJAR experiments
    '''
    def __init__(self, project_hid):
        self.project_hid = project_hid
        self.url = "/experiments"
        super(ExperimentClient, self).__init__()

    def get_experiments(self):
        '''
        Gets all experiments in the project
        '''
        logger.info('Get experiments, project id {}'.format(self.project_hid))
        response = self.request("GET", self.url+'?project_id='+self.project_hid)
        experiments_dict = response.json()
        return [Experiment.from_dict(expt) for expt in experiments_dict]

    def get_experiment(self, experiment_hid):
        '''
        Get details of experiment.
        '''
        logger.info('Get experiment, experiment id {}'.format(experiment_hid))
        try:
            response = self.request("GET", self.url+'/'+experiment_hid)
            return Experiment.from_dict(response.json())
        except NotFoundException:
            return None

    def create_experiment(self, data):
        response = self.request("POST", self.url, data = data)
        if response.status_code != 201:
            raise CreateExperimentException()
        return Experiment.from_dict(response.json())

    def add_experiment_if_not_exists(self, train_dataset, vald_dataset, experiment_title, project_task, \
                                        validation_kfolds, validation_shuffle, \
                                        validation_stratify, validation_train_split, \
                                        algorithms, metric, \
                                        tuning_mode, time_constraint, create_ensemble):
        logger.info('Add experiment if not exists')
        # parameters validation
        # validation with dataset
        if vald_dataset is not None:
            validation = "With dataset"
        else:
            # do train/validation split
            if validation_train_split is not None:
                percents = int(validation_train_split * 100.0)
                validation = "Split {}/{}".format(percents, 100-percents)
            else:
                validation = "{}-fold CV".format(validation_kfolds)

            # shuffle and stratify
            if validation_shuffle:
                validation += ", Shuffle"
            if validation_stratify and project_task == 'bin_class':
                validation += ", Stratify"
            if validation_stratify and project_task != 'bin_class':
                warnings.warn('Cannot use stratify in validation for your project task. Omitting this option in validation.')

        if metric is None or metric == '' or metric not in MLJAR_METRICS:
            metric = MLJAR_DEFAULT_METRICS[project_task]
        if tuning_mode is None or tuning_mode == '' or tuning_mode not in MLJAR_TUNING_MODES:
            tuning_mode = MLJAR_DEFAULT_TUNING_MODE
        if algorithms is None or algorithms == [] or algorithms == '':
            algorithms = MLJAR_DEFAULT_ALGORITHMS[project_task]
        # set default preprocessing if needed
        logger.info('Set default preprocessing')
        dataset_preproc = {}
        if len(train_dataset.column_usage_min['cols_to_fill_na']) > 0:
            dataset_preproc['na_fill'] = 'na_fill_median'
        if len(train_dataset.column_usage_min['cols_to_convert_categorical']) > 0:
            dataset_preproc['convert_categorical'] = 'categorical_to_int'
        # create stub for new experiment
        logger.info('Create new experiment stub')
        expt_params = {
                "train_dataset": {"id": train_dataset.hid, 'title': train_dataset.title},
                "algs":algorithms,
                "preproc": dataset_preproc,
                "single_limit":time_constraint,
                "ensemble":create_ensemble,
                "random_start_cnt": MLJAR_TUNING_MODES[tuning_mode]['random_start_cnt'],
                "hill_climbing_cnt": MLJAR_TUNING_MODES[tuning_mode]['hill_climbing_cnt']
                }
        if vald_dataset is not None:
            expt_params['vald_dataset'] = {"id": vald_dataset.hid, 'title': vald_dataset.title}

        new_expt = Experiment(hid='', title=experiment_title, models_cnt=0, task=project_task,
                                description='', metric=metric, validation_scheme=validation,
                                total_timelog=0, bestalg=[], details={},
                                params=expt_params,
                                compute_now=0, computation_started_at=None, created_at=None,
                                created_by=None, parent_project=self.project_hid)

        # get existing experiments
        experiments = self.get_experiments()
        # check if there are experiments with selected title
        experiments = [e for e in experiments if e.title == new_expt.title]
        # if there are experiments with selected title
        if len(experiments) > 0:
            # check if experiment with the same title has different parameters
            for expt in experiments:
                if not expt.equal(new_expt):
                    print 'The experiment with specified title already exists, but it has different parameters than you specified.'
                    print 'Existing experiment'
                    print str(expt)
                    print 'New experiment'
                    print str(new_expt)
                    print 'Please rename your new experiment with new parameters setup.'
                    return None
            # there is only one experiment with selected title and has the same parameters
            # this is our experiment :)
            if len(experiments) == 1:
                return experiments[0]
            else:
                # there more than 1 experiment, something goes wrong ...
                raise UndefinedExperimentException()
        else:
            # there is no experiment with such title, let's go and create it!
            logger.info('Create new experiment: %s' % new_expt.title)
            # create data for experiment construction by hand
            params = json.dumps(new_expt.params)
            data = {
                'title': new_expt.title,
                'description': '',
                'metric': new_expt.metric,
                'validation_scheme': new_expt.validation_scheme,
                'task': new_expt.task,
                'compute_now': 1,
                'parent_project': self.project_hid,
                'params': params

            }
            return self.create_experiment(data)



        return None
