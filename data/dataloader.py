import torch
#from tfds.features import FeaturesDict



class Set_Dataset(torch.utils.data.Dataset):

	def __init__(self, desired_dataset):
		if desired_dataset == 'CIFAR-10':
			'''
			Dataset_Dict = FeaturesDict({
							'id': Text(shape=(), dtype=tf.string),
							'image': Image(shape=(32, 32, 3), dtype=tf.uint8),
							'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=10)
						   })
			'''
			Dataset_Dict = torch.nn.ModuleDict({
								'id': Text(shape=(), dtype=tf.string),
                                                        	'image': Image(shape=(32, 32, 3), dtype=tf.uint8),
                                                        	'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=10)
							  }, modules=iterable)
		if else desired_dataset == 'CIFAR-100':
			print('not implemented yet')

		if else desired_dataset == 'SVHN':
                        print('not implemented yet')

		else:
			print('Provide with one of the 3 datasets: CIFAR-10, CIFAR-100 or SVHN')






#class Use_Dataset(torch.utils.data.Dataset):
