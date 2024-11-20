import cv2
import numpy as np

def detectField(img):
    imgRed = img[:, :, 0]
    imgGreen = img[:, :, 1]
    imgBlue = img[:, :, 2]
    
    threshold_green = 20  # Umbral para detectar verde
    mask_green = (imgGreen > imgRed + threshold_green) & (imgGreen > imgBlue + threshold_green)
    mask_green = mask_green.astype(np.uint8)  # Convertir a uint8 para OpenCV
    mask_green = cv2.erode(mask_green, np.ones((5, 5), np.uint8), iterations=5)

    # Rellenar huecos usando un cierre morfológico
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))  # Crear un kernel elíptico
    mask_filled = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=5)  # Cierre morfológico

    # Aplicar la máscara rellena a la imagen original
    img_green_filled = img.copy()
    img_green_filled[mask_filled == 0] = [0, 0, 0]
    
    return img_green_filled