from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import cv2

# 验证码中的字符, 就不用汉字了
char_set = '0123456789'
img_height = 32
# img_width = 160
img_channel = 3


# 生成字符对应的验证码
def gen_captcha_text_and_image(captcha_size, char_set=char_set, img_height=img_height):
    # width = captcha_size * 25
    image = ImageCaptcha()
    captcha_text = ''
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text += c
    captcha = image.generate(captcha_text)
    captcha_image = np.array(Image.open(captcha))
    old_h, old_w = captcha_image.shape[:2]
    img_width = int(float(old_w) / old_h * img_height)
    captcha_image = cv2.resize(captcha_image, (img_width, img_height))
    captcha_image = captcha_image.transpose((1, 0, 2))
    return captcha_image, captcha_text


def gen_captcha_text_and_image2(captcha_size, char_set=char_set, img_height=img_height):
    captcha_text = ''
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text += c
    font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
    img_width = captcha_size * 24
    im = np.zeros((img_height, img_width, 3), np.uint8)  # 新建图像，注意一定要是uint8
    img = cv2.putText(im, captcha_text, (0, 28), font, 1.2, (255, 255, 255), 2)  # 添加文字，1.2表示字体大小
    return img, captcha_text


# 转化一个序列列表为稀疏矩阵
def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    return indices, values, shape


# 生成一个训练batch
def get_next_batch(batch_size=128):
    inputs = []
    codes = []
    for i in range(batch_size):
        # 生成不定长度的字串
        image, text = gen_captcha_text_and_image2(4)
        # np.transpose 矩阵转置 (32*256,) => (32,256) => (256,32)
        inputs.append(image)
        codes.append(list(text))

    targets = [np.asarray(i) for i in codes]
    sparse_targets = sparse_tuple_from(targets)
    return inputs, sparse_targets

def gen_captcha_text_and_image2(captcha_size, char_set=char_set, img_height=32):
    captcha_text = ''
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text += c
    font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
    img_width = captcha_size * 24
    im = np.zeros((img_height, img_width, 3), np.uint8)  # 新建图像，注意一定要是uint8
    img = cv2.putText(im, captcha_text, (0, 28), font, 1.2, (255, 255, 255), 2)  # 添加文字，1.2表示字体大小
    return img, captcha_text


def get_next_batch8(batch_size=128):
    inputs = []
    codes = []
    for i in range(batch_size):
        # 生成不定长度的字串
        image, text = gen_captcha_text_and_image2(8)
        cv2.resize(image,(32,256))
        fn = 'C:\\Users\\Lenovo\\Documents\\zhujie\\bb\\'+text+'_'+str(i)+".jpg"
        cv2.imwrite(fn,image)
        print(fn)


if __name__ == '__main__':
    get_next_batch8(3000)
    exit(0)
    # 测试
    inputs, sparse_targets = get_next_batch(2)
    print(sparse_targets)
    cv2.imshow('123', inputs[0])
    cv2.waitKey(0)

   
