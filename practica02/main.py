import functions as f
import cv2
import skimage as ski
import numpy as np
import utils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

def main():
    imgs = os.listdir('practica02/img')
    
    imgs.sort()
    
    #imgs = imgs[len(imgs)//2:]
    #imgs = imgs[len(imgs)//2:]

    for img in imgs:
    #img = mpimg.imread('practica02/img/33.jpg')
        img = mpimg.imread('practica02/img/' + img)
        imgOut = f.detectField(img)
        lines = f.findGrassLines(imgOut)
        playersContours = f.findPlayers(imgOut)
        imgOut = utils.drawGrassLines(imgOut, lines)
        imgOut = utils.drawPlayers(imgOut, playersContours)

        utils.show_imgs([img, imgOut])
        

if __name__ == "__main__":
    main()