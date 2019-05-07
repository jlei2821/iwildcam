import pandas as pd
import numpy as np
import tensorflow as tf
from iwildcam_image_readers import read_zipped_images, read_images

def output_results_from_zip(model, image_shape, data_dir, output_dir):
    
    test = pd.read_csv(str(data_dir)+"/test.csv")

    # Generate split datasets
    print("Loading Images")
    X = read_zipped_images(test['file_name'], str(data_dir)+"/test_images.zip", image_shape,)
    print("Finished...")

    y_pred = model.predict(X, verbose=1,)
    y_pred = np.argmax(y_pred, axis=1).reshape(-1,)
    pd.concat([test['Id'], pd.DataFrame(y_pred.reshape(-1, 1), columns=['Predicted'])], axis=1).to_csv(str(output_dir)+"/predictions.csv", index=False)


def output_results_from_dir(model, image_shape, data_dir, output_dir):
    
    test = pd.read_csv(str(data_dir)+"/test.csv")

    # Generate split datasets
    print("Loading Images")
    X = read_images(test['file_name'], str(data_dir)+"/test_images", image_shape,)
    print("Finished...")

    y_pred = model.predict(X, verbose=1,)
    y_pred = np.argmax(y_pred, axis=1).reshape(-1,)
    pd.concat([test['Id'], pd.DataFrame(y_pred.reshape(-1, 1), columns=['Predicted'])], axis=1).to_csv(str(output_dir)+"/predictions.csv", index=False)