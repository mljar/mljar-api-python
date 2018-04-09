from .base import MljarHttpClient
from ..model.project import Project
from ..exceptions import NotFoundException, CreateProjectException
from ..log import logger

class ProjectClient(MljarHttpClient):
    '''
    Client to interact with MLJAR projects.
    '''
    def __init__(self):
        self.verbose = True
        self.url = "/projects"
        super(ProjectClient, self).__init__()

    def get_projects(self):
        '''
        List all user projects.
        '''
        response = self.request("GET", self.url)
        projects_dict = response.json()
        return [Project.from_dict(proj) for proj in projects_dict]

    def get_project(self, hid):
        '''
        Print out project details and return details in json.
        '''
        try:
            response = self.request("GET", '/'.join([self.url, hid]))
            return Project.from_dict(response.json())
        except NotFoundException:
            return None


    def create_project(self, title, task, description = ''):
        '''
        Creates new project
        '''
        data= {'hardware': 'cloud',
                'scope': 'private',
                'task': task,
                'compute_now': 0,
                'description': description,
                'title':title}
        response = self.request("POST", self.url, data = data)
        if response.status_code != 201:
            raise CreateProjectException()
        return Project.from_dict(response.json())

    def delete_project(self, hid):
        '''
        Deletes project
        '''
        logger.info('Remove project: %s' % hid)
        response = self.request("DELETE", '/'.join([self.url, hid]))
        return response.status_code == 204 or response.status_code == 200

    def create_project_if_not_exists(self, title, task, description = ''):
        '''
        Checks if project with specified title and task exists, if not it adds new project.
        '''
        projects = self.get_projects()
        self.my_project = [p for p in projects if p.title == title and p.task == task]
        # if project with such title does not exist, create one
        if len(self.my_project) == 0:
            self.my_project = self.create_project(title = title,
                                                    description = description,
                                                    task = task)
        else:
            self.my_project = self.my_project[0]

        return self.my_project
