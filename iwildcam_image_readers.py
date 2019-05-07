import numpy as np
import cv2
import zipfile

def read_zipped_images(files, archive, target_size, scale=1., flip_image=False, rotate=0, print_update=10000):
    with zipfile.ZipFile(archive, 'r') as zf:
        imgs = np.empty((len(files),) + target_size + (3,))
        for i, filename in enumerate(files):
            data = zf.read(filename)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
            if flip_image == True:
                img = cv2.flip(img, 1)
            if rotate > 0:
                ctr = tuple(np.array(img.shape[1::-1]) / 2)
                rot = cv2.getRotationMatrix2D(ctr, rotate, 1.0)
                img = cv2.warpAffine(img, rot, img.shape[1::-1], flags=cv2.INTER_LINEAR)
            img = cv2.resize(img, target_size)
            imgs[i, :, :, :] = img * scale
            if (i % print_update) == 0:
                print(f"Loading image {i} of {len(files)}...")
    return imgs


def read_images(files, directory, target_size, scale=1., flip_image=False, rotate=0, print_update=10000):
    imgs = np.empty((len(files),) + target_size + (3,))
    for i, filename in enumerate(files):
        img = cv2.imread(str(directory)+"/"+filename, 1)
        if flip_image == True:
            img = cv2.flip(img, 1)
        if rotate > 0:
            ctr = tuple(np.array(img.shape[1::-1]) / 2)
            rot = cv2.getRotationMatrix2D(ctr, rotate, 1.0)
            img = cv2.warpAffine(img, rot, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        img = cv2.resize(img, target_size)
        imgs[i, :, :, :] = img * scale
        if (i % print_update) == 0:
            print(f"Loading image {i} of {len(files)}...")
    return imgs