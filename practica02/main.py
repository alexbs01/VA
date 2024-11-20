import functions as f
import cv2
import skimage as ski
import numpy as np
import utils
import matplotlib.image as mpimg

def main():
    img = mpimg.imread('practica02/img/100.jpg')
    imgRed = img[:, :, 0]
    imgGreen = img[:, :, 1]
    imgBlue = img[:, :, 2]

    imgRG = np.zeros_like(img)
    imgRG[:, :, 0] = imgRed
    imgRG[:, :, 1] = imgGreen
    
    imgGB = np.zeros_like(img)
    imgGB[:, :, 0] = imgGreen
    imgGB[:, :, 1] = imgBlue
    
    imgBR = np.zeros_like(img)
    imgBR[:, :, 0] = imgBlue
    imgBR[:, :, 1] = imgRed
    
    imgRed_eq = cv2.equalizeHist(imgRed)
    imgGreen_eq = cv2.equalizeHist(imgGreen)
    imgBlue_eq = cv2.equalizeHist(imgBlue)
    
    #img_high_green_erode_dilate = cv2.dilate(img_high_green_erode, np.ones((10, 10), np.uint8), iterations=5)
        
    #utils.show_imgs([imgRed, imgGreen, imgBlue])
    #utils.show_imgs([imgRG, imgGB, imgBR])
    #utils.show_imgs([imgRed_eq, imgGreen_eq, imgBlue_eq])
    #utils.show_imgs([img, img_eq])
    
    imgOut = f.detectField(img)
    utils.show_imgs([img, imgOut])
    
        

if __name__ == "__main__":
    main()