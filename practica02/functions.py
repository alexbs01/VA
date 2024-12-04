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
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_green = np.array([35, 0, 0])
    upper_green = np.array([80, 150, 175])
    
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    
    imgOut = img.copy()
    imgOut[mask_green == 0] = [0, 0, 0]
    #return imgOut
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    imgOut = cv2.erode(imgOut, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    print(kernel)
    imgOut = cv2.morphologyEx(imgOut, cv2.MORPH_CLOSE, kernel, iterations=7)

    imgOut = cv2.cvtColor(imgOut, cv2.COLOR_RGB2GRAY)
    
    #imgOut = cv2.equalizeHist(imgOut)
    #imgOut = cv2.morphologyEx(imgOut, cv2.MORPH_OPEN, kernel, iterations=7)
    #imgOut = cv2.GaussianBlur(imgOut, (75, 75), 0)
    #imgOut = cv2.equalizeHist(imgOut)

    #return imgOut
    #imgOut = cv2.medianBlur(imgOut, 71)
    #imgOut = cv2.equalizeHist(imgOut)
    
    grad_x = cv2.Sobel(imgOut, cv2.CV_64F, 1, 0, ksize=3)  # Gradiente en x
    grad_y = cv2.Sobel(imgOut, cv2.CV_64F, 0, 1, ksize=3)  # Gradiente en y

    # Magnitud del gradiente
    magnitude = cv2.magnitude(grad_x, grad_y)

    # Normalizar la magnitud para mostrarla como imagen
    imgOut = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    imgOut = cv2.equalizeHist(imgOut)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    #imgOut = cv2.morphologyEx(imgOut, cv2.MORPH_CLOSE, kernel, iterations=1)
    imgOut = cv2.medianBlur(imgOut, 9)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 7))
    imgOut = cv2.erode(imgOut, kernel, iterations=1)
    lines = cv2.HoughLinesP(imgOut, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    
    return lines