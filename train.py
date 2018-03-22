"""
Module to train the models.
We just have to train two types of models:
1. ConvNet written from scratch
2. Transfer Learning and Fine Tuning
"""


def train(model, x_train, y_train, nb_epoch=1, ):
    """
    Takes the data


    :param model:
    :param x_train:
    :param y_train:
    :param nb_epoch:
    :return:
    """


    x_val = x_train[10000:]
    y_val = y_train[10000:]

    return model.fit(x_train, y_train, nb_epoch=1, validation_data=(x_val, y_val))

