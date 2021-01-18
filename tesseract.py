'''
Importing required libraries
'''
import os
import numpy as np
from PIL import Image, ImageTk
import PIL
import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, load_model
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from cropyble import Cropyble
import cv2
import shutil
import time
import pytesseract

'''
Enter location of pytesseract (if error msg displayed)
'''
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



'''
Input the image
'''
location = input("Enter the location of the image : ")
image = cv2.imread(location, 0)
thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)[1]

result = cv2.GaussianBlur(thresh, (5, 5), 0)
result = 255 - result

cv2.imwrite('final_img.png', result)

loc = 'final_img.png'
if os.path.isfile(loc):
    print("File exists")

if os.path.isfile('DATA/train/.DS_Store'):
    os.remove('DATA/train/.DS_Store')
if os.path.isfile('DATA/test/.DS_Store'):
    os.remove('DATA/test/.DS_Store')

for i in range(1, 80):
    if os.path.isfile(f'DATA/train/{i}/.DS_Store'):
        os.remove(f'DATA/train/{i}/.DS_Store')
    if os.path.isfile(f'DATA/test/{i}/.DS_Store'):
        os.remove(f'DATA/test/{i}/.DS_Store')

datagen = ImageDataGenerator(rescale=1. / 255,
                             zoom_range=0.2)

trained_image = datagen.flow_from_directory('DATA/train/',
                                            target_size=(32, 32),
                                            class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_image = test_datagen.flow_from_directory('DATA/test/',
                                              target_size=(32, 32),
                                              class_mode='categorical')


def Train_Model():
    '''

    Function to train model using VGG16 weights
    To train the model un-comment the Train_Model() after the function

    '''
    reg = l2(0.010)
    global trained_image, test_image
    model = Sequential()
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    model.add(conv_base)
    model.add(Flatten())
    # model.add(Dense(1024, activation=('relu'), input_dim=512, kernel_regularizer=reg))
    model.add(Dense(512, activation=('relu'), kernel_regularizer=reg))
    model.add(Dropout(.3))
    model.add(Dense(256, activation=('relu'), kernel_regularizer=reg))
    model.add(Dropout(.3))
    model.add(Dense(128, activation=('relu'), kernel_regularizer=reg))
    model.add(Dropout(.2))
    model.add(Dense(79, activation=('softmax')))
    model.summary()
    learn_rate = .001
    sgd = SGD(lr=learn_rate, momentum=.9, nesterov=False)
    # adam = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(trained_image, epochs=30, validation_data=test_image, validation_steps=1)
    model.save('model.h5')


#Un-comment next line to train model
#Train_Model()


def Predict_Model(img):
    '''
    Function to predict the class of the character
    '''
    global trained_image
    new_model = load_model('model.h5')
    img = cv2.resize(img, (32, 32), 3)
    img = np.expand_dims(img, axis=0)
    img = img / 255

    dict_og= {'!': 0, '#': 1, '$': 2, '%': 3, '&': 4, '(': 5, ')': 6, ',': 7, '0': 8, '1': 9, '2': 10, '3': 11, '4': 12, '5': 13,
     '6': 14, '7': 15, '8': 16, '9': 17, ';': 18, '=': 19, '@': 20, 'A': 21, 'B': 22, 'C': 23, 'D': 24, 'E': 25,
     'F': 26, 'G': 27, 'H': 28, 'J': 29, 'K': 30, 'L': 31, 'M': 32, 'N': 33, 'O': 34, 'P': 35, 'Q': 36, 'R': 37,
     'S': 38, 'T': 39, 'U': 40, 'V': 41, 'W': 42, 'Y': 43, 'Z': 44, '[': 45, ']': 46, 'X': 47, '{': 48, '}': 49,
     'u': 50, 'j': 51, 'k': 52, 'r': 53, 's': 54, 't': 55, 'f': 56, '?': 57, 'q': 58, 'l': 59, '.': 60, 'I': 61,
     'g': 62, '̣h': 63, 'x': 64, 'w': 65, 'a': 66, 'b': 67, 'm': 68, 'n': 69, 'p': 70, 'v': 71, 'y': 72, 'z': 73,
     'i': 74, 'o': 75, 'e': 76, 'd': 77, 'c': 78}

    prediction = new_model.predict_classes(img)
    prediction = prediction[0]

    #my_dict = dict(trained_image.class_indices)

    for key, value in dict_og.items():
        if prediction == value:
            return key


def Word_Extract(location):
    """
    Function to extract words from the input image using Cropyble and OpenCV
    """
    if os.path.isdir('WORDS'):
        shutil.rmtree('WORDS')
        os.mkdir('WORDS')
    else:
        os.mkdir('WORDS')

    my_img = Cropyble(location)
    img = PIL.Image.open(location)

    words = my_img.get_words()
    select = []
    for i in range(len(words)):
        if (words[i] != '' or words[i] != ' '):
            if (len(words[i]) > 1):
                select.append(words[i])
            elif (len(words[i]) == 1):
                if (i != 0):
                    if (len(words[i - 1]) > 1 and len(words[i + 1]) > 1):
                        select.append(words[i])
                elif (i == 0):
                    select.append(words[i])

    j = 0

    for i in select:
        rect = my_img.get_box(i)
        crop_img = img.crop((rect[0] - 20, rect[1] - 20, rect[2] + 20, rect[3] + 20))
        crop_img.save(f'WORDS/{j}.png')
        j += 1

    return my_img, select


cropyble_img, select = Word_Extract(loc)

if os.path.isdir('LETTERS'):
    shutil.rmtree('LETTERS')
    os.mkdir('LETTERS')
else:
    os.mkdir('LETTERS')


'''    
Extracting the letters from the WORDS
'''
for i in range(len(os.listdir('WORDS'))):
    os.mkdir(f'LETTERS/{i}')
    image = cv2.imread(f"WORDS/{i}.png")

    edged = cv2.Canny(image, 10, 250)
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    idx = 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if 10 < w and 10 < h:
            idx += 1
            new_img = image[y:y + h, x:x + w]
            cv2.imwrite(f"LETTERS/{i}/" + str(idx) + '.png', new_img)



'''
 Passing each letter to the Predict_Model() and getting output class prediction
'''

text = ''
for i in range(len(os.listdir('LETTERS'))):
    string = ''
    char_dict = {}
    for j in range(len(os.listdir(f'LETTERS/{i}'))):
        img = cv2.imread(f'LETTERS/{i}/{j + 1}.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        char = Predict_Model(img)

        for l in range(len(select[i])):

            if char == select[i][l]:
                char_dict[select[i].index(char)] = char



            else:
                for k in range(len(select[i])):
                    if k != l:
                        if char == select[i][k]:
                            char_dict[k] = char



        keys = list(char_dict.keys())
        if char not in select[i]:
            for val in range(len(select[i])):
                if val not in keys:
                    char_dict[val] = char


    keys = list(char_dict.keys())
    keys.sort()

    for m in keys:
        string += char_dict[m]

    string += ' '
    text += string



'''
Output
'''
for i in range(5):
    print("")
print("The predicted sentence is: ", text)

with open('output.txt', 'w') as file:
    file.write("The predicted sentence is:  ")
    file.write(text)
    file.close()