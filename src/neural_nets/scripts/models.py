from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.initializers import HeNormal

def unet(pretrained_weights= None, input_size = (256,256,1)):
    
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(3, 1, activation = 'softmax')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def resnet101_modfied(input_size = (256,256,1), num_classes = 3):
    assert len(input_size) == 3
    model_resnet  = ResNet101(
        include_top =True,
        weights= None,
        input_shape=input_size, 
        classes=input_size[0]*input_size[1]*num_classes,
        classifier_activation = 'softmax'
    )
    
    
    for layer in model_resnet.layers:
        if isinstance(layer,Conv2D):
            layer.activation = LeakyReLU(alpha=0.01)
            layer.kernel_initializer = HeNormal()
    
    
    predictions = Reshape((input_size[0],input_size[1],3))(model_resnet.layers[-1].output)
    last = Conv2D(3, 1, activation = 'softmax')(predictions)
    modified_model = Model(inputs=model_resnet.input,outputs = last)

    modified_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return modified_model