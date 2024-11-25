import cv2
import numpy as np

def detectField(img):
    imgRed = img[:, :, 0]
    imgGreen = img[:, :, 1]
    imgBlue = img[:, :, 2]
    
    threshold_green = 10
    mask_green = (imgGreen > imgRed + threshold_green) & (imgGreen > imgBlue + threshold_green)
    mask_green = mask_green.astype(np.uint8) * 255
    mask_green = cv2.erode(mask_green, np.ones((5, 5), np.uint8), iterations=5)

    # Rellenar huecos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    mask_filled = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=5)

    img_green_filled = img.copy()
    img_green_filled[mask_filled == 0] = [0, 0, 0]
    
    return img_green_filled

def findPlayers(img):
    imgRed = img[:, :, 0]
    imgGreen = img[:, :, 1]
    imgBlue = img[:, :, 2]
    
    threshold_green = 10
    mask_green = (imgGreen > imgRed + threshold_green) & (imgGreen > imgBlue + threshold_green)
    mask_green = mask_green.astype(np.uint8) * 255
    mask_green = cv2.erode(mask_green, np.ones((3, 3), np.uint8), iterations=9)
    
    img_green_filled = img.copy()
    img_green_filled[mask_green == 0] = [0, 0, 0]

    # Inversa de la máscara
    mask_black = cv2.bitwise_not(mask_green)

    # Detectar contornos de las regiones negras
    contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours
