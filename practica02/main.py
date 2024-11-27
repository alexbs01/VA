import functions as f
import cv2
import skimage as ski
import numpy as np
import utils
import matplotlib.image as mpimg
import os

def main():
    imgs = os.listdir('practica02/img')
    
    imgs.sort()
    
    imgs = imgs[len(imgs)//2:]
    imgs = imgs[len(imgs)//2:]

    for img in imgs:
    #img = mpimg.imread('practica02/img/33.jpg')
        img = mpimg.imread('practica02/img/' + img)
        imgOut = f.detectField(img)
        
        imgOut = f.findPlayers(imgOut)
        #imgOut = utils.drawPlayers(imgOut, playersContours)

        #imgOut = cv2.cvtColor(imgOut, cv2.COLOR_BGR2GRAY)
        #imgOut = cv2.medianBlur(imgWB, 71)

        #imgGray = ski.util.img_as_float(imgGray)
        #imgEqualized = cv2.equalizeHist(imgOut)
        
        #imgEqualized = cv2.equalizeHist(imgOut)
        
        utils.show_imgs([img, imgOut])
    
        

if __name__ == "__main__":
    main()