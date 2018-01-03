import struct
import numpy as np

def read_mnist_data(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

class Dataset:
    def __init__(self, data, labels):
		if isinstance(data, str):
			self.data = read_mnist_data(data)
		else:
			self.data = data
		if isinstance(labels, str):
			self.labels = read_mnist_data(labels)
		else:
			self.labels = labels
    
    def split(self, percentage):
		num_examples = self.data.shape[0]
		indicies = np.random.permutation(num_examples)
		pivot = int(percentage*num_examples)
		left = indicies[:pivot]
		right = indicies[pivot:]
		return Dataset(self.data[left], self.labels[left]), Dataset(self.data[right], self.labels[right])

    def get_epoch_iterator(self, batch_size=32, random=True):
        num_examples = self.data.shape[0]
        ei = np.random.permutation(num_examples) if random else np.arange(num_examples)
        for i in xrange(num_examples/batch_size):
            yield (self.data[i*batch_size:(i+1)*batch_size].reshape(batch_size, -1).T / 127.5 - 1, # [0..255] -> [-1..1]
                   self.labels[i*batch_size:(i+1)*batch_size])
