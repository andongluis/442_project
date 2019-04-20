'''
Functions for graphing accuracy plots and whatnot
'''
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from math import sin, cos, pi
import datetime


def plot_history(adam_hist, sgd_hist=None, path="./", acc=True, loss=False):
    # accs = list(adam_hist.history['acc'])
    # val_accs = list(adam_hist.history['val_acc'])
    losses = list(adam_hist.history['loss'])
    val_losses = list(adam_hist.history['val_loss'])
    if sgd_hist:
        # accs.extend(list(sgd_hist.history['acc']))
        # val_accs.extend(list(sgd_hist.history['val_acc']))
        losses.extend(list(sgd_hist.history['loss']))
        val_losses.extend(list(sgd_hist.history['val_loss']))

    # Plot training & validation accuracy values
    # plt.figure(1)
    # plt.plot(accs)
    # plt.plot(val_accs)
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.savefig(path + "accs.png",  bbox_inches='tight')

    # Plot training & validation loss values
    plt.figure(2)
    plt.plot(losses)
    plt.plot(val_losses)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.savefig(path + "losses.png",  bbox_inches='tight')


def _process_nums(str_in):
    if isinstance(str_in, str):
        str_in = str_in.replace(",","")
        if "m" in str_in:
            str_in = str_in.split("m")[0]
            return float(str_in)*1000000
        elif "k" in str_in:
            str_in = str_in.split("k")[0]
            return float(str_in)*1000
    return float(str_in)


def process_raw_csv(csv_file, drop_cols=["url","shortcode","username"]):

    df = pd.read_csv(csv_file, skiprows=[0])
    kept_cols = ['shortcode', 'username', 'followers', 'following', 'posts', 'hastags/text',
                 'timestamp of pic', 'timestamp when scrapped', 'url', 'likes', 'name']
    df = df[kept_cols]

    # Drop NaN rows
    df = df.dropna()

    # Drop drop_cols
    df = df.drop(drop_cols, axis=1)

    # Process numbers
    num_cols = ["followers", "following", "likes", "posts"]
    for col in num_cols:
        df[col] = df[col].apply(_process_nums)

    # Process timestamp
    time_cols = ["timestamp of pic", "timestamp when scrapped"]
    for col in time_cols:
        df[col] = df[col].apply(lambda x: x.strip(",")[0] if isinstance(x, str) else x).astype(int)
        df[col] = df[col].apply(datetime.datetime.fromtimestamp)
        year_col = col + "_year"
        df[col] = pd.DatetimeIndex(df[col])
        df[year_col] = pd.DatetimeIndex(df[col]).year
        
        month_col = col + "_month"
        df[month_col + "_sin"] = df[col].apply(lambda x: sin(2*x.month*pi/(12)))
        df[month_col + "_cos"] = df[col].apply(lambda x: cos(2*pi*x.month/(12)))

        day_col = col + "_day"
        df[day_col + "_sin"] = df[col].apply(lambda x: sin(2*pi*x.day/(x.daysinmonth)))
        df[day_col + "_cos"] = df[col].apply(lambda x: cos(2*pi*x.day/(x.daysinmonth)))

        dayw_col = col + "_dayofweek"
        df[dayw_col + "_sin"] = df[col].apply(lambda x: sin(2*pi*x.dayofweek/(6)))
        df[dayw_col + "_cos"] = df[col].apply(lambda x: cos(2*pi*x.dayofweek/(6)))

        hr_col = col + "_hr"
        df[hr_col + "_sin"] = df[col].apply(lambda x: sin(2*pi*x.hour/(24)))
        df[hr_col + "_cos"] = df[col].apply(lambda x: cos(2*pi*x.hour/(24)))

    return df


def baseline(trainY, y_mean=-1, name="Train"):
    if y_mean == -1:
        y_mean = np.mean(trainY)
    train_mse = np.mean( np.absolute( trainY - y_mean ) )
    print("{} baseline average MAE: {}".format(name, round(train_mse, 2)))
    # trainY = np.reshape(trainY, trainY.shape[0])
    # percent = np.mean(np.absolute( (trainY - y_mean) / trainY ))*100
    # print("{} baseline average MAPE: {}%".format(name, round(percent, 2 )))
    return y_mean
