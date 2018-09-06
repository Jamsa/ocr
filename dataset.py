import os
import numpy as np
from PIL import Image

DICT = '0123456789'


class DataSet(object):
    """数据集类"""

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
        """生成数据集文件和标签列表"""
        if len(self.__files) > 0:
            return self.__files, self.__labels

        file_list = os.listdir(self.rootdir)

        for line in file_list:
            name = line.split("_")
            # 标签取文件名以下划线分割的第一段，标签长度应该为8
            if len(name) > 1 and len(name[0]) == 8 and line.endswith(".jpg"):
                self.__files.append(self.rootdir + '/' + line)
                self.__labels.append(list(name[0])) # 将标签转化为字符序列
                self.__encoded_labels.append(list(name[0]))

        return self.__files, self.__labels, self.__encoded_labels

    def get_next_batch(self, batch_size=64, gray_scale=True, transpose=True, resize_to=(256, 32)):
        """从数据集中随机获取一批样本"""
        sel = np.random.choice(len(self.__labels), batch_size)
        images = []
        for i, file in enumerate(np.array(self.__files)[sel]):
            image = Image.open(file)
            if gray_scale:
                image = image.convert('L')  # 转为灰度图
            if resize_to is not None:
                image = image.resize(resize_to)
            # if i == 0:
            #     image.show()
            image = np.asarray(image)
            if transpose:
                image = np.transpose(image)

            images.append(image)
        labels = np.array(self.__encoded_labels)[sel]
        # sparse_labels = [np.asarray(i) for i in labels]
        sparse_labels = sparse_tuple_from(labels)

        images = np.array(images)
        return images, sparse_labels, labels, (np.ones(batch_size) * images.shape[0])


def sparse_tuple_from(sequences, dtype=np.int32):
    """
    转化一个序列列表为稀疏矩阵，针对一个batch的数据

    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    # n行数据，每行的数据为seq，即每个元素都是字符的序列，在 DataSet.__gen_dataset 中处理过
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))  # 每个元素都变为 [行号,列号] 索引表示
        values.extend(seq) # sparse 矩阵的 values

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    #shape为(64,8) max(0)取最大行，即(63,7)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


if __name__ == '__main__':
    #print(sparse_tuple_from([[0,0,0,1,3]]))
    #exit(0)

    dataset = DataSet('../dataline')
    (files, el, lb, l) = dataset.get_next_batch()
    print(el)
    print(lb[0])