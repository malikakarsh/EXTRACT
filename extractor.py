import os
import PIL
import cv2
import shutil
from cropyble import Cropyble




def Word_Extract(location):
    
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
                if ( i != 0):
                    if (len(words[i-1]) > 1 and len(words[i+1]) > 1):
                        select.append(words[i])
                elif ( i == 0):
                    select.append(words[i])

                    
    j = 0

    for i in select:
        rect = my_img.get_box(i)
        crop_img = img.crop((rect[0]-20,rect[1]-20,rect[2]+20,rect[3]+20))
        crop_img.save(f'WORDS/{j}.png')
        j += 1
    
    return my_img, select



def Letter_Extract():
    if os.path.isdir('LETTERS'):
        shutil.rmtree('LETTERS')
        os.mkdir('LETTERS')
    else:
        os.mkdir('LETTERS')
    
    
    for i in range(len(os.listdir('WORDS'))):
        os.mkdir(f'LETTERS/{i}')
        image = cv2.imread(f"WORDS/{i}.png") 
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
        edged = cv2.Canny(image, 10, 250) 
        (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        idx = 0 
        for c in cnts: 
            x,y,w,h = cv2.boundingRect(c) 
            if w>50 and h>50: 
                idx+=1 
                new_img=image[y:y+h,x:x+w] 
                cv2.imwrite(f"LETTERS/{i}/" + str(idx) + '.png', new_img) 
