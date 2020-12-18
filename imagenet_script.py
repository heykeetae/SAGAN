import os
import pickle
from PIL import Image
import numpy as np
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

for j in range(2,11):
    data_path = 'data/imagenet/train/train_data_batch_{}'.format(j)


def numpytoimage(data_path='data/imagenet/train/train_data_batch_1', output_dir ='data/imagenet/train'):
    d = unpickle(data_path)
    x = d['data']
    y = d['labels']
    mean_image = d['mean']
    img_size = int((x.shape[1]/3)**(0.5))
    data_size = x.shape[0]
    img_size2 = img_size**2
    x = np.dstack((x[:,:img_size2], x[:,img_size2:2*img_size2], x[:,2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size,3))
    # saving the image
    for i in range(data_size):
        class_count = y[i]
        folder_path = output_dir+ '/{}'.format(class_count)
        if not (os.path.isdir(folder_path)):
            os.mkdir(folder_path)
        file_count = len(os.listdir(folder_path))
        file_name = output_dir+'/{}/{}.png'.format(class_count, file_count+1)
        image = x[i,:,:,:]
        image = (255.0 / image.max() * (image - image.min())).astype(np.uint8)
        im = Image.fromarray(image)
        im.save(file_name)
        if i%10000 == 0:
            print(file_name)
    print(data_path)

def main():
    for i in range(1,11):
        data_path = 'data/imagenet/train/train_data_batch_{}'.format(i)
        final_dir = 'data/imagenet/train'
        numpytoimage(data_path, final_dir)

if __name__ == "__main__":
    main()
