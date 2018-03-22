"""
This module provides the following models:
    1. Pre-trained models:
        Resnet50
        InceptionNet
        Xception
        Vgg16
    2. Transfer learning and fine tuning:
        transfer_xception
    3. Make your own CovNet model from scratch
"""
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications import xception
from keras.applications import inception_v3
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras.models import Model, Input

class Models(object):
    """
    Defines all the models
    """
    def __init__(self):
        pass

    class PreTrained(object):

        def __init__(self):
            pass

        def ResNet50():
            """
            Args: None

            Returns: A pre-trained ResNet50 model which is ready to predict
            """
            ResNet50_model = ResNet50(weights='imagenet')
            return ResNet50_model

        def vgg16():
            """
            Args: None

            Returns: A pre-trained (on imagenet) VGG16 model which is ready to predict
            """
            vgg = VGG16(weights='imagenet')
            return vgg

        def XceptionNet():
            """
            Args: None

            Returns: A pre-trained (on imagenet) Xception model which is ready to predict
            """
            xception_net = xception.Xception(weights='imagenet')
            return xception_net

        def Inception_V3():
            """
            Args: None

            Returns: A pre-trained (on imagenet) Inception_V3 model which is ready to predict
            """
            inception = inception_v3.InceptionV3(weights='imagenet')
            return inception

    def myConvNet(self):
        """
        Args: None

        Returns a custom made ConvNet model.
        This can be trained from scratch.
        """
        model = Sequential()

        model.add(BatchNormalization(input_shape=(224, 224, 3)))
        model.add(Conv2D(filters=16, kernel_size=3, activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(BatchNormalization())

        model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(BatchNormalization())

        model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(BatchNormalization())

        model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(BatchNormalization())

        model.add(GlobalAveragePooling2D())
        model.add(Dense(120, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def TransferFine(self, top_two_layers=False, top_layers=False):
        """
        This models does Transfer Learning and fine tunes
        a network. We use XCeption network here.
        For Transfer Learning: Add spatial average pooling layer
                            followed by FC and then softmax layer to predict the
                            120 classes

        For Fine Tuning: There are two options as described below

        Args: top_two_layers: fine tunes the top two layers
              top_layers: fine tunes the top 116 layers
              Set either of the two to be true

        Returns: Fine tuned model weight some weights transferred and some layers fine tuned.
        """

        base_model = xception.Xception(weights='imagenet', include_top=False)

        x = base_model.output
        x = BatchNormalization()(x)
        x = GlobalAveragePooling2D()(x)

        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)

        predictions = Dense(120, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        optimizer = RMSprop(lr=0.001, rho=0.9)

        if top_two_layers is True:
            for layer in base_model.layers:
                layer.trainable = False

        if top_layers is True:
            for layer in model.layers[:116]:
                layer.trainable = False
            for layer in model.layers[116:]:
                layer.trainable = True

        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=["accuracy"])

        return model


if __name__ == '__main__':
    model = Models()
    yo = model.TransferFine(top_layers=True)
    yo.summary()
    print(yo)