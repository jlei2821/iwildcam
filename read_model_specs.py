import tensorflow as tf

from iwildcam_models import *

if __name__ == '__main__':

    image_shape = (64, 64)

    models = [tf_cnn_model, tf_cnn_model_pool]
    run_names = ["testrun",]

    for i, model in enumerate(models):

        print(f"Model: {model} summary")

        # Instantiate Model Object
        m = model(image_shape + (3,))

        m.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        # # Print Model Execution Summary
        m.summary()