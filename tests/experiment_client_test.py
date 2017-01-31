'''
ProjectClient tests.
'''
import os
import unittest

from mljar.client.project import ProjectClient
from mljar.client.experiment import ExperimentClient

class ExperimentClientTest(unittest.TestCase):

    def test_create_and_delete(self):
        proj_title = 'Test project'
        expt_title = 'Test experiment'

        pc = ProjectClient()
        new_project = pc.create_project(title = proj_title)
        ec = ExperimentClient(new_project.hid)
        self.assertNotEqual(ec, None)
