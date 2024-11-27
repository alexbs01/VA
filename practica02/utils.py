import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

def maskField(image):
    # Convertir al espacio de color HSV para segmentar el verde
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Rango dinámico para el color verde
    lower_green = np.array([35, 20, 20])  # Tonalidad, saturación, brillo
    upper_green = np.array([85, 255, 255])
    
    # Crear una máscara para el color verde
    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)

    # Aplicar operaciones morfológicas para limpiar la máscara
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask_cleaned = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel, iterations=3)
    
    # Opcional: dilatar ligeramente para aumentar las áreas verdes
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask_final = cv2.dilate(mask_cleaned, kernel_dilate, iterations=1)
    
    contours = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    maxContour = max(contours, key=cv2.contourArea)
    
    contourMask = np.zeros_like(mask_final)
    cv2.drawContours(contourMask, [maxContour], -1, 255, -1)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    contourMask = cv2.morphologyEx(contourMask, cv2.MORPH_CLOSE, kernel, iterations=5)
    
    return contourMask

def drawPlayers(img, contours):
    imgOut = img.copy()
    height, width = img.shape[:2]
    print(height, width)
    print("###")
    
    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        print(w, h)
        if w < h * 1.3 and h < 3 * w and  w < 500 and  h < 200:
            cv2.rectangle(imgOut, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    return imgOut

def show_imgs(arrayImages):
    _, axs = plt.subplots(len(arrayImages), 1, figsize=(24, 16))

    if len(arrayImages) == 1:
        axs = [axs]

    for i, image in enumerate(arrayImages):
        axs[i].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        axs[i].axis('off')
        

    plt.tight_layout()
    plt.show()