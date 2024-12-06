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

def lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    """
        Verifica si dos líneas se cortan
    """
    def ccw(ax, ay, bx, by, cx, cy):
        return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

    return (
        ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4)
        and ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4)
    )

def drawGrassLines(img, lines):
    imgOut = img.copy()
    angleThreshold = 20
    minSeparation = 75

    filteredLines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            if y2 < y1:
                x1, y1, x2, y2 = x2, y2, x1, y1

            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            if angle < 0:
                angle += 180

            if 0 <= angle < angleThreshold or 180 - angleThreshold < angle <= 180:
                continue

            lineLength = np.hypot(x2 - x1, y2 - y1)

            tooClose = False
            for fx1, fy1, fx2, fy2, fangle, flength in filteredLines:

                dist = min(
                    np.hypot(x1 - fx1, y1 - fy1),
                    np.hypot(x2 - fx2, y2 - fy2),
                    np.hypot(x1 - fx2, y1 - fy2),
                    np.hypot(x2 - fx1, y2 - fy1),
                )
                
                if dist < minSeparation:
                    if abs(fangle - 90) < abs(angle - 90) or flength > lineLength:
                        tooClose = True
                        break
                    else:
                        filteredLines.remove((fx1, fy1, fx2, fy2, fangle, flength))
                        break

            if tooClose:
                continue

            intersects = 0
            for fx1, fy1, fx2, fy2, _, _ in filteredLines:
                if lines_intersect(x1, y1, x2, y2, fx1, fy1, fx2, fy2):
                    intersects += 1
                    if intersects > 1:
                        break

            if intersects > 1:
                continue

            filteredLines.append((x1, y1, x2, y2, angle, lineLength))

        # Dibujar las líneas seleccionadas
        for x1, y1, x2, y2, _, _ in filteredLines:
            cv2.line(imgOut, (x1, y1), (x2, y2), (255, 0, 255), 2)

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