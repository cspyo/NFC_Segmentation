##패키지 등록
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


## 데이터 불러오기
dir_data = './datasets'

##
nframe_train = 5
nframe_val = 2
nframe_test = 2

dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)



for i in range(0, 5):

    name_label = './label/%02d.jpg' % i
    name_input = './input/%02d.jpg' % i

    img_label = Image.open(os.path.join(dir_data, name_label))
    img_input = Image.open(os.path.join(dir_data, name_input))

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)


for i in range(5, 7):

    name_label = './label/%02d.jpg' % i
    name_input = './input/%02d.jpg' % i

    img_label = Image.open(os.path.join(dir_data, name_label))
    img_input = Image.open(os.path.join(dir_data, name_input))

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)


for i in range(7, 9):

    name_label = './label/%02d.jpg' % i
    name_input = './input/%02d.jpg' % i

    img_label = Image.open(os.path.join(dir_data, name_label))
    img_input = Image.open(os.path.join(dir_data, name_input))

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)



##
plt.subplot(121)
plt.imshow(label_, cmap='gray')
plt.title('label')

plt.subplot(122)
plt.imshow(input_, cmap='gray')
plt.title('input')

plt.show()