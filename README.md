# EXTRACT

## Introduction

EXTRACT is an optical character recognition engine for various operating systems which extracts texts from an image and converts them to plain text.

This model is a very primitive form of the original google tesseract which extracts texts from an image and converts them to plain text.

## Modules/Library REQUIREMENTS:

  1) os
  2) numpy
  3) PIL
  4) sys
  5) keras
  6) cropyble
  7) cv2
  8) shutil
  
## Features
 
a)  Extracts text from input image
b) Works on lowercase,uppercase, number ans special characters.
c) Saves the output in output.txt to allow search.

## How To Run the script:

NOTE1:- The trained model is not provided. So for the very first time run the script as it is. Once the model is trained:
                                          COMMENT OUT 'Train_Model()' then run the script for further use.
                                          

![](sentences/sentence_format.png)

Run the script on your terminal: 'python3 tesseract.py':
input image is:
![](sentences/say.png)

output is (the predicted result is at the bottom):
![](sentences/terminal_output2.png)

The input image can be of any number of words example:
![](sentences/say3.png)

output is:
![](sentences/terminal_output.png)


## Contributors

- Akarsh Malik
- Angad Ripudaman Singh Bajwa

## Future Work

1)  To add characters of your own, make sure to add them in the train and test dataset
2) Change the output of the softmax layer in Train_Model function to the total number of trained characters.
2) Re-train the model
3) Test your image