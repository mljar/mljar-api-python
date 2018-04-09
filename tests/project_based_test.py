'''
ProjectBasedTest tests.
'''
import os
import unittest
import pandas as pd
import sys

from mljar.client.project import ProjectClient

def get_postfix():
    return '-v-'+str(sys.version_info.major)

class ProjectBasedTest(unittest.TestCase):

    @staticmethod
    def clean_projects():
        project_client = ProjectClient()
        projects = project_client.get_projects()
        for proj in projects:
            if proj.title.startswith('Test'):
                project_client.delete_project(proj.hid)

    @classmethod
    def setUpClass(cls):
        ProjectBasedTest.clean_projects()

    #@classmethod
    #def tearDownClass(cls):
    #    ProjectBasedTest.clean_projects()
