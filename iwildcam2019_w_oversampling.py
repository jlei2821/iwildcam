# Import Standard Modules
import pandas as pd
import numpy as np
from pathlib import Path
import random
from sklearn.metrics import f1_score, precision_score, recall_score

# Import Tensorflow Modules
import tensorflow as tf
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, InputLayer, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing import image



# Import Custom Callbacks
from custom_callbacks import ScoreCallback, TimerCallback


if __name__ == '__main__':

    # Set Paths
    projpath = Path("C:/Users/alexa/OneDrive/repositories/iwildcam2019")
    trainpath = Path("C:/kaggle_4")
    testpath = Path("D:/kaggle_4_data")

    # Set Configurations
    seed = 42
    epochs = 50
    num_classes = 23
    batch_size = 256
    image_shape = (64, 64)
    kernel_size = (3, 3)
    strides = (2, 2)
    run_name = "run01"
    sample_training_set = True
    n_samples = 150000

    # Define Callbacks
    es = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=5,
        verbose=1,
        mode='auto'
    )

    tb = TensorBoard(
        log_dir=projpath/f"logs/{run_name}", 
        histogram_freq=0, 
        write_graph=True, 
        write_images=False
    )

    mc = ModelCheckpoint(
        filepath=str(projpath)+'/{epoch:02d}-{val_loss:.2f}-{val_acc:.4f}.hdf5',
        monitor='val_acc',
        verbose=1,
        save_best_only=True
    )

    sc = ScoreCallback()

    tm = TimerCallback()

    # Set Seed for Reproducibility
    np.random.seed(seed)

    # Import Training Set
    train = pd.read_csv(trainpath/"train.csv")

    # Generate one hot encoded labels and reattach to main dataframe
    y_one_hot = pd.DataFrame(to_categorical(train['category_id']))

    train_one_hot = pd.concat([train, y_one_hot], axis=1)

    if sample_training_set == True:
        train_cat = [train_one_hot[train_one_hot['category_id']==i] for i in range(23)]

        new_training_set = []

        for cat in train_cat:
            if len(cat) == 0:
                new_training_set.append(pd.DataFrame([]))
            else:
                if len(cat) < n_samples:
                    while len(cat) < n_samples:
                        cat = pd.concat([cat, cat], axis=0)
                
                new_training_set.append(cat)

        train_new = pd.concat(new_training_set, axis=0)

        train_one_hot = train_new.iloc[random.sample(range(len(train_new)), n_samples)]

        print(np.bincount(np.array(train_new['category_id']))/len(train_new))

    # Assign class weights for cost function
    cls_wgts = np.bincount(np.array(train['category_id']))/len(train)

    # Create data generators to pipe data into the model
    train_gen = ImageDataGenerator(rescale=1./255, validation_split=.20, horizontal_flip=True, rotation_range=0)
    train_set = train_gen.flow_from_dataframe(dataframe=train_one_hot, directory=str(trainpath/"train_images"), x_col='file_name', y_col=[x for x in range(23)], class_mode='other', target_size=image_shape, batch_size=batch_size, subset='training', seed=seed, drop_duplicates=False)
    valid_set = train_gen.flow_from_dataframe(dataframe=train_one_hot, directory=str(trainpath/"train_images"), x_col='file_name', y_col=[x for x in range(23)], class_mode='other', target_size=image_shape, batch_size=batch_size, subset='validation', shuffle=False, drop_duplicates=False)

    # Initialize sequential model
    m = Sequential()

    # For now, using strides of 2 instead of max pooling to dimensionality reduction
    m.add(Conv2D(filters=64, kernel_size=kernel_size, strides=strides, padding='same', input_shape=image_shape+(3,)))
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

    m.add(Conv2D(filters=512, kernel_size=kernel_size, strides=strides, padding='same'))
    m.add(Activation('relu'))
    m.add(BatchNormalization())
    # m.add(Dropout(0.1))

    # m.add(Conv2D(filters=512, kernel_size=kernel_size, strides=strides, padding='same'))
    # m.add(Activation('relu'))
    # m.add(BatchNormalization())
    # # m.add(Dropout(0.1))

    # m.add(Conv2D(filters=512, kernel_size=kernel_size, strides=strides, padding='same'))
    # m.add(MaxPooling2D((2, 2)))
    # m.add(Activation('relu'))
    # # m.add(Dropout(0.1))

    m.add(Flatten())

    m.add(BatchNormalization())
    m.add(Dense(128))
    m.add(Activation('relu'))
    # m.add(Dropout(0.5))

    m.add(BatchNormalization())
    m.add(Dense(23, activation='softmax', activity_regularizer=l2(0.01)))

    m.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Generate Model Summary
    m.summary()

    # Fit the model
    m.fit_generator(
        train_set, 
        steps_per_epoch=train_set.n//train_set.batch_size,
        validation_data=valid_set, 
        validation_steps=valid_set.n//valid_set.batch_size,
        epochs=epochs, 
        verbose=1, 
        # class_weight=cls_wgts, 
        max_queue_size=batch_size, 
        callbacks=[es, tb, tm, mc],
        # workers=4,
        # use_multiprocessing=True
    )

    y_val_pred = m.predict_generator(valid_set, verbose=1,)
    y_val_pred = y_val_pred.argmax(axis=1)

    val_obs = len(y_val_pred)

    y_val_act = np.array([]).reshape(-1, 23)
    for batch in valid_set:
        y_val_act = np.vstack((y_val_act, batch[1])) 
        if len(y_val_act) >= val_obs:
            break

    y_val_act = y_val_act.argmax(axis=1)

    f1 = f1_score(y_val_act, y_val_pred, average='macro')
    prec = precision_score(y_val_act, y_val_pred, average='macro')
    rec = recall_score(y_val_act, y_val_pred, average='macro')

    print(f"F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

    test = pd.read_csv(testpath/"test.csv")

    test_gen = ImageDataGenerator(rescale=1./255)
    test_set = test_gen.flow_from_dataframe(dataframe=test, directory=str(testpath/"test_images"), x_col='file_name', y_col=None, class_mode=None, target_size=image_shape, batch_size=batch_size, shuffle=False)

    y_pred = m.predict_generator(test_set, verbose=1,)
    y_pred = np.argmax(y_pred, axis=1).reshape(-1,)
    pd.concat([test['Id'], pd.DataFrame(y_pred.reshape(-1, 1), columns=['Predicted'])], axis=1).to_csv(projpath/"predictions.csv", index=False)


