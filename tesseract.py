

## importing the modules
from  input import GiveInput
from preprocessing import processing
from model_dev import Train_Model
from extractor import Word_Extract , Letter_Extract
from prediction import Result


if __name__ == "__main__":
    
    
    loc = GiveInput()
    trained_image , test_image = processing()
    Train_Model()
    cropyble_img,select = Word_Extract(loc)
    Letter_Extract()
    text = Result(select)
    for i in range(10):
        print("")
    print("The predicted sentence is: ",text)
    
