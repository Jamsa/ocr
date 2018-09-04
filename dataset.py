import os
import numpy as np
from PIL import Image

DICT = '0123456789'


class DataSet(object):

    def __init__(self, rootdir):
        self.__files = []
        self.__labels = []
        self.__encoded_labels = []
        self.rootdir = rootdir

        self.dict = {}

        for i, char in enumerate(DICT):
            self.dict[char] = i + 1

        self.__gen_dataset()

    def __gen_dataset(self):
        if len(self.__files) > 0:
            return self.__files, self.__labels

        file_list = os.listdir(self.rootdir)

        for line in file_list:
            name = line.split("_")
            if len(name) > 1 and len(name[0]) == 8 and line.endswith(".jpg"):
                self.__files.append(self.rootdir + '/' + line)
                self.__labels.append(list(name[0]))
                self.__encoded_labels.append(list(name[0]))

        return self.__files, self.__labels, self.__encoded_labels

    def get_next_batch(self, batch_size=64):
        sel = np.random.choice(len(self.__labels), batch_size)
        images = []
        for i, file in enumerate(np.array(self.__files)[sel]):
            image = Image.open(file)
            image = image.convert('L')
            image = image.resize((256, 32))
            # if i == 0:
            #     image.show()
            images.append(np.transpose(np.asarray(image)))
        labels = np.array(self.__encoded_labels)[sel]
        sparse_labels = [np.asarray(i) for i in labels]
        sparse_labels = sparse_tuple_from(labels)

        return np.array(images), sparse_labels, labels, (np.ones(batch_size) * 256)


#转化一个序列列表为稀疏矩阵
def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


if __name__ == '__main__':
    #print(sparse_tuple_from([[0,0,0,1,3]]))
    #exit(0)

    dataset = DataSet('/Users/zhujie/Documents/devel/python/keras/chinese-ocr-chinese-ocr-python-3.6/train/data/dataline')
    (files, el, lb, l) = dataset.get_next_batch()
    print(el)
    print(lb[0])