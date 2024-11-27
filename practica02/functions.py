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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask_players = cv2.dilate(mask_players, kernel, iterations=3)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask_players = cv2.morphologyEx(mask_players, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_players = cv2.morphologyEx(mask_players, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask_players
    # Detectar contornos de las regiones no verdes (jugadores)
    contours, _ = cv2.findContours(mask_players, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours
