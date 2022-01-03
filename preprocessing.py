import os
from keras.preprocessing.image import ImageDataGenerator
def processing():
    if os.path.isfile('DATA/train/.DS_Store'):
        os.remove('DATA/train/.DS_Store')
    if os.path.isfile('DATA/test/.DS_Store'):
        os.remove('DATA/test/.DS_Store')
        
    for i in range(1,80):
        if os.path.isfile(f'DATA/train/{i}/.DS_Store'):
            os.remove(f'DATA/train/{i}/.DS_Store')
        if os.path.isfile(f'DATA/test/{i}/.DS_Store'):
            os.remove(f'DATA/test/{i}/.DS_Store')
    
    datagen = ImageDataGenerator(rescale = 1./255,
                                zoom_range = 0.2)
    
    
    trained_image = datagen.flow_from_directory('DATA/train',
                                                target_size = (32,32),
                                                class_mode = 'categorical')
    
    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    test_image = test_datagen.flow_from_directory('DATA/test',
                                                target_size = (32,32),
                                                class_mode = 'categorical')
    return trained_image , test_image
    


