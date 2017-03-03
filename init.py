import pickle
import numpy as np

def unpickle(file):
	with open(file, 'rb') as f:
		return pickle.load(f, fix_imports=True, encoding="bytes")

data_file_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
ori_data = [unpickle(f) for f in data_file_list]

data = [d[b'data'] for d in ori_data]
labels = [d[b'labels'] for d in ori_data]

in_size = 3 * 32 * 32
out_size = 10

def gen_xy(count):
	'''生成训练数据'''
	ret_x = np.ndarray(shape=(count, in_size))
	ret_y = np.ndarray(shape=(count, out_size))
	for _ in range(0, count):
		#TODO
		pass
	return ret_x, ret_y