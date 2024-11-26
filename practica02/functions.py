import cv2
import numpy as np
import utils

def detectField(img):
    mask_filled = utils.maskField(img)
    #return mask_filled
    img_green_filled = img.copy()
    img_green_filled[mask_filled == 0] = [0, 0, 0]
    
    return img_green_filled

def findPlayers(img):
    imgRed = img[:, :, 0]
    imgGreen = img[:, :, 1]
    imgBlue = img[:, :, 2]
    
    threshold_green = 0
    mask_green = (imgGreen > imgRed + threshold_green) & (imgGreen > imgBlue + threshold_green)
    mask_green = mask_green.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 3))
    mask_green = cv2.erode(mask_green, np.ones((7, 7), np.uint8), iterations=4)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    img_green_filled = img.copy()
    img_green_filled[mask_green == 0] = [0, 0, 0]

    # Inversa de la máscara
    mask_black = cv2.bitwise_not(mask_green)

    # Detectar contornos de las regiones negras
    contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours
