import cv2
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import Dense,Flatten
from keras.models import Sequential, load_model


def Train_Model():
    global trained_image, test_image
    model = Sequential()
    conv_base = VGG16(weights='imagenet', include_top = False, input_shape = (32,32,3))
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(26,activation = 'softmax'))
    model.compile(optimizer= 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.fit(trained_image, epochs = 30, validation_data = test_image, validation_steps = 1)
    model.save('model.h5')
    



def Predict_Model(img):
    global trained_image
    new_model = load_model('model.h5')
    img = cv2.resize(img,(32,32),3)
    img = np.expand_dims(img,axis=0)
    img = img / 255
    letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','1','2','3','4','5','6','7','8','9','0',',',';',':','?','!','.','@','#','$','%','&','(',')','{','}','[',']']
    
    prediction = new_model.predict_classes(img)
    prediction = prediction[0]
    
    my_dict = dict(trained_image.class_indices)
    
    for key,value in my_dict.items():
        if prediction == value:
            return key



