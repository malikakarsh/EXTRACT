import cv2
def GiveInput():
    location = input("Enter the location of the image : ")
    image = cv2.imread(location,0)
    thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)[1]

    result = cv2.GaussianBlur(thresh, (5,5), 0)
    result = 255 - result

    cv2.imwrite('final_img.png',result)
    loc = 'final_img.png'
    return loc

