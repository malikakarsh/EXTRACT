import cv2
import os
from model_dev import Predict_Model


def Result(select):
    text = ''
    for i in range(len(os.listdir('LETTERS'))):
        string = ''
        char_dict = {}
        for j in range(len(os.listdir(f'LETTERS/{i}'))):
            img = cv2.imread(f'LETTERS/{i}/{j+1}.png')
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
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
        #print(keys)
        
        for m in keys:
            string += char_dict[m]
            
            
        string += ' '
        text += string
    return text;