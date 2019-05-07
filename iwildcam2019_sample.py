# Import Standard Modules
import pandas as pd
import numpy as np
from pathlib import Path
import random
from sklearn.metrics import f1_score, precision_score, recall_score
import cv2
import zipfile

# Import Tensorflow Modules
import tensorflow as tf
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, InputLayer, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing import image

# Import Custom Callbacks
# from custom_callbacks import ScoreCallback, TimerCallback


def read_zipped_images(files, archive, target_size, flip_image=False):
    with zipfile.ZipFile(archive, 'r') as zf:
        imgs = np.empty((len(files),) + target_size + (3,), dtype='uint8')
        for i, filename in enumerate(files):
            data = zf.read(filename)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
            img = cv2.resize(img, target_size)
            if flip_image == True:
                img = cv2.flip(img, 1)
            imgs[i, :, :, :] = img
    return imgs


if __name__ == '__main__':

    # Set Paths
    p = Path("D:/kaggle_4_data")
    # testpath = Path("D:/kaggle_4_data")

    # Set Configurations
    seed = 42
    epochs = 20
    num_classes = 23
    batch_size = 128
    image_shape = (32, 32)
    kernel_size = (3, 3)
    strides = (2, 2)
    run_name = "run01"
    sample_training_set = True
    n_samples = 1000

    # Define Callbacks
    # es = EarlyStopping(
    #     monitor='val_loss',
    #     min_delta=0,
    #     patience=3,
    #     verbose=1,
    #     mode='auto'
    # )

    # tb = TensorBoard(
    #     log_dir=projpath/f"logs/{run_name}", 
    #     histogram_freq=0, 
    #     write_graph=True, 
    #     write_images=False
    # )

    # mc = ModelCheckpoint(
    #     filepath=str(projpath)+'/{epoch:02d}-{val_loss:.2f}-{val_acc:.4f}.hdf5',
    #     monitor='val_acc',
    #     verbose=1,
    #     save_best_only=True
    # )

    # sc = ScoreCallback()

    # tm = TimerCallback()

    # Set Seed for Reproducibility
    np.random.seed(seed)

    # Import Training Set
    train = pd.read_csv(p/"train.csv")

    # Assign class weights for cost function
    cls_wgts = np.bincount(np.array(train['category_id']))/len(train)


    # Generate one hot encoded labels and reattach to main dataframe
    y_one_hot = pd.DataFrame(to_categorical(train['category_id']))

    train_one_hot = pd.concat([train, y_one_hot], axis=1)

    if sample_training_set == True:
        train_one_hot = train_one_hot.iloc[random.sample(range(len(train_one_hot)), n_samples)]


    train_data = read_zipped_images(train_one_hot['file_name'], p/"train_images.zip", image_shape,)
    train_labels = to_categorical(np.array(train_one_hot['category_id']), num_classes=23)


    # Initialize sequential model
    m = Sequential()

    # For now, using strides of 2 instead of max pooling to dimensionality reduction
    m.add(Conv2D(filters=32, kernel_size=kernel_size, strides=strides, padding='same', input_shape=image_shape+(3,)))
    m.add(Activation('relu'))
    m.add(BatchNormalization())
    # m.add(Dropout(0.1))

    m.add(Conv2D(filters=64, kernel_size=kernel_size, strides=strides, padding='same'))
    m.add(Activation('relu'))
    m.add(BatchNormalization())
    # m.add(Dropout(0.1))

    m.add(Conv2D(filters=128, kernel_size=kernel_size, strides=strides, padding='same'))
    m.add(Activation('relu'))
    m.add(BatchNormalization())
    # m.add(Dropout(0.1))

    m.add(Conv2D(filters=256, kernel_size=kernel_size, strides=strides, padding='same'))
    m.add(Activation('relu'))
    m.add(BatchNormalization())
    # m.add(Dropout(0.1))

    m.add(Conv2D(filters=256, kernel_size=kernel_size, strides=strides, padding='same'))
    m.add(Activation('relu'))
    m.add(BatchNormalization())
    # m.add(Dropout(0.1))

    m.add(Conv2D(filters=256, kernel_size=kernel_size, strides=strides, padding='same'))
    m.add(Activation('relu'))
    # m.add(Dropout(0.1))

    m.add(Flatten())

    m.add(BatchNormalization())
    m.add(Dense(2304))
    m.add(Activation('relu'))
    m.add(Dropout(0.5))

    m.add(BatchNormalization())
    m.add(Dense(23, activation='softmax'))

    m.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Generate Model Summary
    m.summary()

    # Fit the model
    m.fit(
        train_data, train_labels,
        batch_size=32,
        validation_split=.25, 
        epochs=epochs, 
        verbose=1, 
        class_weight=cls_wgts, 
        # callbacks=[es, tb, tm, mc],
        # workers=4,
        # use_multiprocessing=True
    )

    # y_val_pred = m.predict(valid_set, verbose=1,)
    # y_val_pred = y_val_pred.argmax(axis=1)

    # val_obs = len(y_val_pred)

    # y_val_act = np.array([]).reshape(-1, 23)
    # for batch in valid_set:
    #     y_val_act = np.vstack((y_val_act, batch[1])) 
    #     if len(y_val_act) >= val_obs:
    #         break

    # y_val_act = y_val_act.argmax(axis=1)

    # f1 = f1_score(y_val_act, y_val_pred, average='macro')
    # prec = precision_score(y_val_act, y_val_pred, average='macro')
    # rec = recall_score(y_val_act, y_val_pred, average='macro')

    # print(f"F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

    # test = pd.read_csv(testpath/"test.csv")

    # test_gen = ImageDataGenerator(rescale=1./255)
    # test_set = test_gen.flow_from_dataframe(dataframe=test, directory=str(testpath/"test_images"), x_col='file_name', y_col=None, class_mode=None, target_size=image_shape, batch_size=batch_size, shuffle=False)

    # y_pred = m.predict_generator(test_set, verbose=1,)
    # y_pred = np.argmax(y_pred, axis=1).reshape(-1,)
    # pd.concat([test['Id'], pd.DataFrame(y_pred.reshape(-1, 1), columns=['Predicted'])], axis=1).to_csv(projpath/"predictions.csv", index=False)


