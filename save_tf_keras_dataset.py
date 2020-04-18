'''
  Tensorflow 2.1 + Python3.6環境にて動作確認
'''
import os
import numpy as np
from PIL import Image

from tensorflow import keras
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.datasets import imdb, reuters
from tensorflow.keras.datasets import boston_housing

def save_cifar10():
    OUT_DIR = 'cifar10'

    # Load data from keras API
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # define class names from ex: https://www.cs.toronto.edu/~kriz/cifar.html
    class_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # make train/test dirs and class dirs
    for cid, class_name in enumerate(class_list):
        os.makedirs(os.path.join(OUT_DIR,'train', '{:d}_{}'.format(cid,class_name)), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR,'test', '{:d}_{}'.format(cid,class_name)), exist_ok=True)

    # convert train data
    for num, (x_data, y_data) in enumerate(zip(x_train, y_train)):
        cid = y_data[0] # y_data is an 1 element array
        # make file path
        fpath = os.path.join(OUT_DIR,'train', '{:d}_{}'.format(cid,class_list[cid]), 'train_{:05d}.png'.format(num))
        # convert numpy to pillow and save
        Image.fromarray(x_data).save(fpath)

    # convert test data
    for num, (x_data, y_data) in enumerate(zip(x_test, y_test)):
        cid = y_data[0] # y_data is an 1 element array
        # make file path
        fpath = os.path.join(OUT_DIR,'test', '{:d}_{}'.format(cid,class_list[cid]), 'test_{:05d}.png'.format(num))
        # convert numpy to pillow and save
        Image.fromarray(x_data).save(fpath)

    print()
    print('Saved to '+OUT_DIR+'/')
    print()

def save_cifar100():
    OUT_DIR = 'cifar100'

    # Load data from keras API
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    # define class names from ex: https://github.com/keras-team/keras/issues/2653
    class_list = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

    # make train/test dirs and class dirs
    for cid, class_name in enumerate(class_list):
        os.makedirs(os.path.join(OUT_DIR,'train', '{:02d}_{}'.format(cid,class_name)), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR,'test', '{:02d}_{}'.format(cid,class_name)), exist_ok=True)

    # convert train data
    for num, (x_data, y_data) in enumerate(zip(x_train, y_train)):
        cid = y_data[0] # y_data is an 1 element array
        # make file path
        fpath = os.path.join(OUT_DIR,'train', '{:02d}_{}'.format(cid,class_list[cid]), 'train_{:05d}.png'.format(num))
        # convert numpy to pillow and save
        Image.fromarray(x_data).save(fpath)

    # convert test data
    for num, (x_data, y_data) in enumerate(zip(x_test, y_test)):
        cid = y_data[0] # y_data is an 1 element array
        # make file path
        fpath = os.path.join(OUT_DIR,'test', '{:02d}_{}'.format(cid,class_list[cid]), 'test_{:05d}.png'.format(num))
        # convert numpy to pillow and save
        Image.fromarray(x_data).save(fpath)

    print()
    print('Saved to '+OUT_DIR+'/')
    print()

def save_mnist():
    OUT_DIR = 'mnist'

    # Load data from keras API
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # make train/test dirs and class dirs
    for cid in range(10):
        os.makedirs(os.path.join(OUT_DIR,'train', '{:d}'.format(cid)), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR,'test', '{:d}'.format(cid)), exist_ok=True)

    # convert train data
    for num, (x_data, y_data) in enumerate(zip(x_train, y_train)):
        # make file path
        fpath = os.path.join(OUT_DIR,'train', '{:d}'.format(y_data), 'train_{:05d}.png'.format(num))
        # convert numpy to pillow and save
        Image.fromarray(x_data).save(fpath)

    # convert test data
    for num, (x_data, y_data) in enumerate(zip(x_test, y_test)):
        # make file path
        fpath = os.path.join(OUT_DIR,'test', '{:d}'.format(y_data), 'test_{:05d}.png'.format(num))
        # convert numpy to pillow and save
        Image.fromarray(x_data).save(fpath)

    print()
    print('Saved to '+OUT_DIR+'/')
    print()

def save_fasion_mnist():
    OUT_DIR = 'fasion_mnist'

    # Load data from keras API
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # define class names from ex: https://www.tensorflow.org/tutorials/keras/classification
    class_list = ['T-shirt_top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle_boot']

    # make train/test dirs and class dirs
    for cid, class_name in enumerate(class_list):
        os.makedirs(os.path.join(OUT_DIR,'train', '{:d}_{}'.format(cid,class_name)), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR,'test', '{:d}_{}'.format(cid,class_name)), exist_ok=True)

    # convert train data
    for num, (x_data, y_data) in enumerate(zip(x_train, y_train)):
        # make file path
        fpath = os.path.join(OUT_DIR,'train', '{:d}_{}'.format(y_data,class_list[y_data]), 'train_{:05d}.png'.format(num))
        # convert numpy to pillow and save
        Image.fromarray(x_data).save(fpath)

    # convert test data
    for num, (x_data, y_data) in enumerate(zip(x_test, y_test)):
        # make file path
        fpath = os.path.join(OUT_DIR,'test', '{:d}_{}'.format(y_data,class_list[y_data]), 'test_{:05d}.png'.format(num))
        # convert numpy to pillow and save
        Image.fromarray(x_data).save(fpath)

    print()
    print('Saved to '+OUT_DIR+'/')
    print()

def save_IMDB():
    OUT_DIR = 'IMDB'

    # Load data from keras API
    (x_train, y_train), (x_test, y_test) = imdb.load_data()

    # get word index
    word_index = imdb.get_word_index()
    # make dictionary to reference index
    word_list = {(value+3):key for key,value in word_index.items()}
    # define invalid string to remove them later
    INVALID_STR = '#$%'
    word_list[0] = INVALID_STR
    word_list[1] = INVALID_STR
    word_list[2] = INVALID_STR

    # define class names 0:negative / 1:positive
    class_list = ['negative','positive']

    # make train/test dirs and class dirs
    for cid, class_name in enumerate(class_list):
        os.makedirs(os.path.join(OUT_DIR,'train', '{:d}_{}'.format(cid,class_name)), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR,'test', '{:d}_{}'.format(cid,class_name)), exist_ok=True)

    # convert train data
    for num, (x_data, y_data) in enumerate(zip(x_train, y_train)):
        # make file path
        fpath = os.path.join(OUT_DIR,'train', '{:d}_{}'.format(y_data,class_list[y_data]), 'train_{:05d}.txt'.format(num))
        with open(fpath, mode='w', encoding='utf-8') as f:
            # convert indices and join words with space
            word_org = ' '.join(word_list[inx] for inx in x_data )
            # remove invalid strings
            word_org = word_org.replace(INVALID_STR+' ', '')
            # save text
            f.write(word_org)

    # convert test data
    for num, (x_data, y_data) in enumerate(zip(x_test, y_test)):
        # make file path
        fpath = os.path.join(OUT_DIR,'test', '{:d}_{}'.format(y_data,class_list[y_data]), 'test_{:05d}.txt'.format(num))
        with open(fpath, mode='w', encoding='utf-8') as f:
            # convert indices and join words with space
            word_org = ' '.join(word_list[inx] for inx in x_data )
            # remove invalid strings
            word_org = word_org.replace(INVALID_STR+' ', '')
            # save text
            f.write(word_org)

    print()
    print('Saved to '+OUT_DIR+'/')
    print()

def save_reuters():
    OUT_DIR = 'reuters'

    # Load data from keras API
    (x_train, y_train), (x_test, y_test) = reuters.load_data()

    # get word index
    word_index = reuters.get_word_index()
    # make dictionary to reference index
    word_list = {(value+3):key for key,value in word_index.items()}
    INVALID_STR = '#$%'
    # define invalid string to remove them later
    word_list[0] = INVALID_STR
    word_list[1] = INVALID_STR
    word_list[2] = INVALID_STR

    # define class names from ex: https://github.com/keras-team/keras/issues/12072
    class_list = ['cocoa','grain','veg-oil','earn','acq','wheat','copper','housing','money-supply',
        'coffee','sugar','trade','reserves','ship','cotton','carcass','crude','nat-gas',
        'cpi','money-fx','interest','gnp','meal-feed','alum','oilseed','gold','tin',
        'strategic-metal','livestock','retail','ipi','iron-steel','rubber','heat','jobs',
        'lei','bop','zinc','orange','pet-chem','dlr','gas','silver','wpi','hog','lead']

    # make train/test dirs and class dirs
    for cid, class_name in enumerate(class_list):
        os.makedirs(os.path.join(OUT_DIR,'train', '{:02d}_{}'.format(cid,class_name)), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR,'test', '{:02d}_{}'.format(cid,class_name)), exist_ok=True)

    # convert train data
    for num, (x_data, y_data) in enumerate(zip(x_train, y_train)):
        # make file path
        fpath = os.path.join(OUT_DIR,'train', '{:02d}_{}'.format(y_data,class_list[y_data]), 'train_{:05d}.txt'.format(num))
        with open(fpath, mode='w', encoding='utf-8') as f:
            # convert indices and join words with space
            word_org = ' '.join(word_list[inx] for inx in x_data )
            # remove invalid strings
            word_org = word_org.replace(INVALID_STR+' ', '')
            # save text
            f.write(word_org)

    # convert test data
    for num, (x_data, y_data) in enumerate(zip(x_test, y_test)):
        # make file path
        fpath = os.path.join(OUT_DIR,'test', '{:02d}_{}'.format(y_data,class_list[y_data]), 'test_{:05d}.txt'.format(num))
        with open(fpath, mode='w', encoding='utf-8') as f:
            # convert indices and join words with space
            word_org = ' '.join(word_list[inx] for inx in x_data )
            # remove invalid strings
            word_org = word_org.replace(INVALID_STR+' ', '')
            # save text
            f.write(word_org)

    print()
    print('Saved to '+OUT_DIR+'/')
    print()

def save_boston_housing():
    OUT_DIR = 'boston_housing'
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data from keras API
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()

    # define feature names from https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html
    feature_list = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

    # convert train data
    with open(os.path.join(OUT_DIR, 'train_data.csv'), mode='w', encoding='utf-8') as f:
        # write header
        f.write(','.join(feature_list)+'\n')
        # write feature data
        for x_data,y_data in zip(x_train, y_train):
            f.write(','.join(map(str,np.append(x_data, y_data)))+'\n')

    # convert test data
    with open(os.path.join(OUT_DIR, 'test_data.csv'), mode='w', encoding='utf-8') as f:
        # write header
        f.write(','.join(feature_list)+'\n')
        # write feature data
        for x_data,y_data in zip(x_test, y_test):
            f.write(','.join(map(str,np.append(x_data, y_data)))+'\n')

    print()
    print('Saved to '+OUT_DIR+'/')
    print()



def ask_command():
    print('0: end')
    print('1: Save CIFAR10 images')
    print('2: Save CIFAR100 images')
    print('3: Save MNIST images')
    print('4: Save Fasion-MNIST images')
    print('5: Save IMDB Review Texts')
    print('6: Save Reuters Topics Texts')
    print('7: Save Boston Housing data CSV')
    command = input('select menu -> ')

    val = -1
    if command.isdigit():
        val = int(command)

    if val >= 0 and val <=7:
        return val
    else:
        print('invalid command!')
        print()
        ask_command()


if __name__ == '__main__':
    val = -1
    while val != 0:
        val = ask_command()
        if val == 1:
            save_cifar10()
        elif val == 2:
            save_cifar100()
        elif val == 3:
            save_mnist()
        elif val == 4:
            save_fasion_mnist()
        elif val == 5:
            save_IMDB()
        elif val == 6:
            save_reuters()
        elif val == 7:
            save_boston_housing()

        print()
