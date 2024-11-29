import cv2
import numpy as np
import utils

def detectField(img):
    mask_filled = utils.maskField(img)

    img_green_filled = img.copy()
    img_green_filled[mask_filled == 0] = [0, 0, 0]
    
    return img_green_filled

def findPlayers(image):
    img = cv2.GaussianBlur(image, (5, 5), 0)
    # Convertir la imagen al espacio de color HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Rango de valores para el color verde (puedes ajustarlo según el campo)
    lower_green = np.array([35, 20, 20])  # Mínimo para verde
    upper_green = np.array([85, 255, 255])  # Máximo para verde
    
    # Crear una máscara para el color verde
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Invertir la máscara para detectar regiones no verdes (potenciales jugadores)
    mask_players = cv2.bitwise_not(mask_green)
    
    # Aplicar operaciones morfológicas para limpiar el ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_players = cv2.dilate(mask_players, kernel, iterations=5)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_players = cv2.morphologyEx(mask_players, cv2.MORPH_OPEN, kernel, iterations=3)
    mask_players = cv2.morphologyEx(mask_players, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Detectar contornos de las regiones no verdes (jugadores)
    contours, _ = cv2.findContours(mask_players, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def findGrassLines(image):
    img = image.copy()
    # Convertir la imagen al espacio de color HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Rango de valores para el color verde (puedes ajustarlo según el campo)
    lower_green = np.array([35, 0, 0])  # Mínimo para verde
    upper_green = np.array([80, 150, 165])  # Máximo para verde
    
    # Crear una máscara para el color verde
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #mask_green = cv2.erode(mask_green, kernel=kernel, iterations=1)
    
    imgOut = img.copy()
    imgOut[mask_green == 0] = [0, 0, 0]
    #return imgOut
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    print(kernel)
    #imgOut = cv2.dilate(imgOut, kernel, iterations=59)
    #imgOut = cv2.erode(imgOut, kernel, iterations=59)
    
    imgOut[:, :, 1] = cv2.GaussianBlur(imgOut[:, :, 1], (25, 25), 0)
    imgOut[:, :, 1] = cv2.equalizeHist(imgOut[:, :, 1])
    imgOut[:, :, 1] = cv2.morphologyEx(imgOut[:, :, 1], cv2.MORPH_CLOSE, kernel, iterations=3)
    imgOut[:, :, 1] = cv2.dilate(imgOut[:,:,1], kernel, iterations=3)
    imgOut[:, :, 1] = cv2.erode(imgOut[:,:,1], kernel, iterations=3)
    #imgOut = cv2.GaussianBlur(imgOut, (7, 7), 0)
    imgOut = cv2.Canny(imgOut, 15, 30)
    imgOut = cv2.HoughLines(imgOut, 1, 2, 2)
    #imgOut = cv2.GaussianBlur(imgOut, (7, 7), 0)
    
    #imgOut = cv2.Canny(imgOut, 1, 10)
    
    
    return imgOut
