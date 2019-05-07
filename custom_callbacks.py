from tensorflow.keras.callbacks import Callback
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.utils import to_categorical
import datetime

class ScoreCallback(Callback):
    
    def on_train_begin(self, logs={}):
        self.val_f1 = []
        self.val_precision = []
        self.val_recall = []

    def on_epoch_end(self, epoch, logs={}):
        X = self.validation_data[0]
        y = self.validation_data[1]
        y_pred = self.model.predict(X)

        y_pred_argmax = to_categorical(y_pred.argmax(axis=1))

        f1 = f1_score(y, y_pred_argmax, average='macro')
        prec = precision_score(y, y_pred_argmax, average='macro')
        rec = recall_score(y, y_pred_argmax, average='macro')

        self.val_f1.append(f1)
        self.val_precision.append(prec)
        self.val_recall.append(rec)

        print(f"val_f1: {f1:.4f} - val_prec: {prec:.4f} - val_rec: {rec:.4f}")

        return


class TimerCallback(Callback):

    def on_train_begin(self, logs=None):
        self.start_time = datetime.datetime.now()
        print(f"Training Start: {self.start_time}")

    def on_train_end(self, logs=None):
        self.end_time = datetime.datetime.now()
        total_duration = self.end_time - self.start_time
        print(f"Training End: {self.end_time} - Total Duration: {total_duration.total_seconds()}s")
