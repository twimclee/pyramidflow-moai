import os
import yaml

class MPathFileManager:
	def __init__(self, volume, project, subproject, task, version):
		self.volume = volume
		self.project = project
		self.subproject = subproject
		self.task = task
		self.version = version

		self.project_path = None
		self.subproject_path = None
		self.task_path = None
		self.version_path = None
		self.weight_path = None
		self.train_result = None
		self.test_dataset = None
		self.test_result = None
		self.train_path = None
		self.val_path = None

		self.hyp_yaml = None
		self.data_yaml = None
		self.result_csv = None

		self.__make_dir()

	def __make_dir(self):
		# self.project_path = f'/{self.volume}/{self.project}'
		self.project_path = f'../{self.project}'
		if not os.path.exists(self.project_path):
		     os.makedirs(self.project_path)

		self.subproject_path = f'{self.project_path}/{self.subproject}'
		if not os.path.exists(self.subproject_path):
		     os.makedirs(self.subproject_path)

		self.task_path = f'{self.subproject_path}/{self.task}'
		if not os.path.exists(self.task_path):
		     os.makedirs(self.task_path)

		self.version_path = f'{self.task_path}/{self.version}'
		if not os.path.exists(self.version_path):
		     os.makedirs(self.version_path)

		self.weight_path = f'{self.version_path}/weights'
		if not os.path.exists(self.weight_path):
		     os.makedirs(self.weight_path)

		self.train_result = f'{self.version_path}/training_result'
		if not os.path.exists(self.train_result):
		     os.makedirs(self.train_result)

		self.test_dataset = f'{self.version_path}/inference_dataset'
		if not os.path.exists(self.test_dataset):
		     os.makedirs(self.test_dataset)

		self.test_result = f'{self.version_path}/inference_result'
		if not os.path.exists(self.test_result):
			os.makedirs(self.test_result)

		self.train_path = f'{self.task_path}/train_dataset/train'
		self.val_path = f'{self.task_path}/train_dataset/valid'

		self.hyp_yaml = f'{self.task_path}/train_dataset/hyp.yaml'
		self.data_yaml = f'{self.task_path}/train_dataset/data.yaml'
		self.result_csv = f'{self.train_result}/results.csv'

		self.test_hyp_yaml = f'{self.train_result}/hyp.yaml'
		self.test_data_yaml = f'{self.train_result}/data.yaml'

	def load_train_hyp(self, mcls):
		self.__load_yaml(self.hyp_yaml, mcls)

	def load_train_data(self, mcls):
		self.__load_yaml(self.data_yaml, mcls)

	def load_test_hyp(self, mcls):
		self.__load_yaml(self.test_hyp_yaml, mcls)

	def load_test_data(self, mcls):
		self.__load_yaml(self.test_data_yaml, mcls)

	def __load_yaml(self, path, mcls):
		with open(path, 'r') as file:
			config = yaml.safe_load(file)
        
		for key, value in config.items():
			setattr(mcls, key, value)


	def save_hyp(self, mcls):
		self.__save_yaml(f'{self.train_result}/hyp.yaml', mcls)

	def save_data(self, mcls):
		self.__save_yaml(f'{self.train_result}/data.yaml', mcls)

	def __save_yaml(self, path, mcls):
		with open(path, 'w') as f:
			yaml.dump(mcls.__dict__, f, default_flow_style=False, allow_unicode=True)

