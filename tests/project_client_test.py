'''
ProjectClient tests.
'''
import os
import unittest

from mljar.client.project import ProjectClient
class ProjectClientTest(unittest.TestCase):

    def test_create_and_delete(self):
        '''
        Get list of projects, add new project, again get lists of projects and
        compare if new list length is greater than old one.
        '''
        proj_title = 'Test project-01'
        pc = ProjectClient()
        projects_before = pc.get_projects()
        new_project = pc.create_project(title = proj_title)
        self.assertEqual(new_project.title, proj_title)
        projects_after = pc.get_projects()
        self.assertEqual(len(projects_before) + 1, len(projects_after))
        pc.delete_project(new_project.hid)
        projects_after = pc.get_projects()
        self.assertEqual(len(projects_before), len(projects_after))


    def test_project_get(self):
        '''
        Test project get method.
        '''
        proj_title = 'Test project-02'
        pc = ProjectClient()
        projects_before = pc.get_projects()
        new_project = pc.create_project(title = proj_title)
        project = pc.get_project(hid = new_project.hid)
        self.assertEqual(new_project.hid, project.hid)
        self.assertEqual(new_project.title, project.title)
        self.assertEqual(new_project.task, project.task)
        self.assertEqual(new_project.scope, project.scope)
        self.assertEqual(new_project.hardware, project.hardware)
        pc.delete_project(new_project.hid)
        project = pc.get_project(hid = new_project.hid)
        self.assertEqual(project, None)

    def test_project_get_unknown_hid(self):
        '''
        Test invalid hid value in project get method.
        '''
        pc = ProjectClient()
        project = pc.get_project(hid = 'invalid_hid_value')
        self.assertEqual(project, None)

    def test_create_if_not_exists(self):
        pass
