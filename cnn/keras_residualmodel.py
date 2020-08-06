import keras
from keras.models import Model, Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Convolution2D, Input,Activation, ZeroPadding2D, MaxPooling2D, Flatten, Add
from keras.optimizers import SGD
from keras.objectives import sparse_categorical_crossentropy as scc


#https://www.kaggle.com/yadavsarthak/residual-networks-and-mnist


def make_residual_model(img_size, num_channels, num_classes):
    # In order to make things less confusing, all layers have been declared first, and then used
    
    # declaration of layers
    input_img = Input((img_size, img_size, num_channels), name='input_layer')
    zeroPad1 = ZeroPadding2D((1,1), name='zeroPad1', )
    zeroPad1_2 = ZeroPadding2D((1,1), name='zeroPad1_2', )
    layer1 = Convolution2D(6, 3, 3, subsample=(2, 2), name='major_conv')
    layer1_2 = Convolution2D(16, 3, 3, subsample=(2, 2),  name='major_conv2')
    zeroPad2 = ZeroPadding2D((1,1), name='zeroPad2', )
    zeroPad2_2 = ZeroPadding2D((1,1), name='zeroPad2_2')
    layer2 = Convolution2D(6, 3, 3, subsample=(1,1), name='l1_conv')
    layer2_2 = Convolution2D(16, 3, 3, subsample=(1,1), name='l1_conv2')


    zeroPad3 = ZeroPadding2D((1,1), name='zeroPad3')
    zeroPad3_2 = ZeroPadding2D((1,1), name='zeroPad3_2')
    layer3 = Convolution2D(6, 3, 3, subsample=(1, 1), name='l2_conv')
    layer3_2 = Convolution2D(16, 3, 3, subsample=(1, 1), name='l2_conv2')

    layer4 = Dense(64, activation='relu', name='dense1')
    layer5 = Dense(16, activation='relu', name='dense2')

    final = Dense(num_classes, activation='softmax', name='classifier')
    
    # declaration completed
    
    first = zeroPad1(input_img)
    second = layer1(first)
    second = BatchNormalization(name='major_bn')(second)
    second = Activation('relu', name='major_act')(second)

    third = zeroPad2(second)
    third = layer2(third)
    third = BatchNormalization(name='l1_bn')(third)
    third = Activation('relu', name='l1_act')(third)

    third = zeroPad3(third)
    third = layer3(third)
    third = BatchNormalization(name='l1_bn2')(third)
    third = Activation('relu', name='l1_act2')(third)

    res = Add(name='res')([third, second])
    # model.add(Merge([left, right]))
    # res = merge([third, second], mode='sum', name='res')


    first2 = zeroPad1_2(res)
    second2 = layer1_2(first2)
    second2 = BatchNormalization(name='major_bn2')(second2)
    second2 = Activation('relu', name='major_act2')(second2)


    third2 = zeroPad2_2(second2)
    third2 = layer2_2(third2)
    third2 = BatchNormalization(name='l2_bn')(third2)
    third2 = Activation('relu', name='l2_act')(third2)

    third2 = zeroPad3_2(third2)
    third2 = layer3_2(third2)
    third2 = BatchNormalization(name='l2_bn2')(third2)
    third2 = Activation('relu', name='l2_act2')(third2)

    res2 = Add(name='res2')([third2, second2])

    # res2 = merge([third2, second2], mode='sum', name='res2')

    res2 = Flatten()(res2)

    res2 = layer4(res2)
    res2 = Dropout(0.4, name='dropout1')(res2)
    res2 = layer5(res2)
    res2 = Dropout(0.4, name='dropout2')(res2)
    res2 = final(res2)
    model = Model(input=input_img, output=res2)
    
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=scc,
                       optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), #keras.optimizers.Adadelta(),
                    #  optimizer=keras.optimizers.Adadelta(), 
                      metrics=['accuracy'])




    # sgd = SGD(decay=0., lr=0.01, momentum=0.9, nesterov=True)
    # model.compile(loss=scc, optimizer=sgd, metrics=['accuracy'])
    return model