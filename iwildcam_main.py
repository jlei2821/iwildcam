# Import Standard Modules
import numpy as np
import pandas as pd
from pathlib import Path
import random
from sklearn.metrics import f1_score, precision_score, recall_score

# Import Tensorflow Modules
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.models import save_model

# Import Custom Modules
from custom_callbacks import ScoreCallback, TimerCallback
from iwildcam_models import tf_cnn_model
from iwildcam_image_readers import read_images
from iwildcam_output_results import output_results_from_dir

def main(proj_dir, data_dir):

    # Define Project and Data Paths
    projpath = Path(proj_dir)
    datapath = Path(data_dir)

    # Set Configurations
    seed = 42
    epochs = 100
    num_classes = 23
    batch_size = 1000
    image_shape = (64, 64)
    sample_training_set = False
    n_samples = 1000
    testset_size = .20

    # Set Model Run Parameters
    run_models = [tf_cnn_model,]
    run_names = ["testrun",]

    # Define Static Callbacks
    print("Defining Callbacks...")
    es = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=5,
        verbose=1,
        mode='auto'
    )

    sc = ScoreCallback()
    tm = TimerCallback()

    np.random.seed(seed)

    # Import Training Set
    print("Importing Dataset...")
    train_df = pd.read_csv(datapath/"train.csv")

    # Assign class weights for cost function
    cls_wgts = np.bincount(np.array(train_df['category_id']))/len(train_df)

    if sample_training_set == True:
        # Keep from trying to pull more samples than the length of the dataset
        n_samples = min(n_samples, len(train_df))

        # Select records by row index
        train_df = train_df.iloc[random.sample(range(len(train_df)), n_samples)]

    # Split dataset into training and testing sets
    train_df['dataset'] = 'train'
    train_df.iloc[random.sample(range(len(train_df)), round(len(train_df)*testset_size)), -1] = 'test'
    train, test = train_df[train_df['dataset'] == 'train'], train_df[train_df['dataset'] == 'test']

    # Generate split datasets
    print("Splitting and loading dataset...")
    X_train = read_images(train['file_name'], datapath/"train_images", image_shape,)
    print("X_train finished...")

    y_train = to_categorical(np.array(train['category_id']), num_classes=num_classes)
    print("y_train finished...")

    X_test = read_images(test['file_name'], datapath/"train_images", image_shape,)
    print("X_test finished...")

    y_test = to_categorical(np.array(test['category_id']), num_classes=num_classes)
    print("X_test finished...")

    print("Done loading data, ready for modeling...")

    best_f1 = -float('Inf')

    for i, model in enumerate(run_models):

        print(f"Starting Model {i+1}: {model}")

        # Setting Callbacks with Variable Path Outputs 
        tb = TensorBoard(
            log_dir=projpath/f"logs/{run_names[i]}", 
            histogram_freq=0, 
            write_graph=True, 
            write_images=False
        )

        mc = ModelCheckpoint(
            filepath=str(projpath)+f"/models/{run_names[i]}_best.hdf5",
            monitor='val_acc',
            verbose=1,
            save_best_only=True
        )

        # Instantiate Model Object
        m = model(image_shape + (3,))

        m.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        # # Print Model Execution Summary
        m.summary()

        # # Fit Model on Training Data
        m.fit(
            X_train, y_train,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            epochs=epochs, 
            verbose=1, 
            class_weight=cls_wgts, 
            callbacks=[es, tb, tm, mc],
        )

        save_model(m, str(projpath)+f"/models/{run_names[i]}.hdf5")

        y_pred = m.predict(X_test)
        y_pred_argmax = to_categorical(y_pred.argmax(axis=1), num_classes=num_classes)

        f1 = f1_score(y_test, y_pred_argmax, average='macro')
        prec = precision_score(y_test, y_pred_argmax, average='macro')
        rec = recall_score(y_test, y_pred_argmax, average='macro')

        print(f"val_f1: {f1:.4f} - val_prec: {prec:.4f} - val_rec: {rec:.4f}")

        if f1 > best_f1:
            best_model = m
            if len(run_models) > 1:
                print("Saving Best Model...")
                save_model(m, str(projpath)+f"/models/best_model.hdf5")
                best_f1 = f1

    return best_model, image_shape

if __name__ == '__main__':

    # Set if you want to output results
    output = True
    proj_dir = "C:/Users/alexa/OneDrive/repositories/iwildcam2019" # Path where code files are located
    data_dir = "D:/kaggle_4_data" # Path where the data files are located

    m, image_shape = main(proj_dir, data_dir)
    if output == True:
        output_results_from_dir(m, image_shape, data_dir, proj_dir)
