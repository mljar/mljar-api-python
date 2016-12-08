from client import MljarClient

mljar_client = MljarClient(token = 'your_token')
#mljar_client.get_projects()
#mljar_client.get_project_details('YPVd39E43ODn')

#mljar_client.create_project(title="MLJAR-API-2", description='Auto', task='bin_class')


fname = '/home/piotr/webs/mljar/test/data/binary_part_iris_converted.csv'
mljar_client.add_new_dataset(project_hid='pWovmM0N3YEy', title='train', file_path=fname)

#mljar_client.get_projects()
