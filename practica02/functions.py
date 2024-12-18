import cv2
import numpy as np
import utils

def detectField(img):
    mask_filled = utils.maskField(img)

    img_green_filled = img.copy()
    img_green_filled[mask_filled == 0] = [0, 0, 0]
    
    return img_green_filled

def findPlayers(image):
    mask_players = utils.maskPlayers(image)
    # return mask_players
    contours, _ = cv2.findContours(mask_players, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def findGrassLines(image):
    imgOut = np.copy(image)

    imgOut = cv2.medianBlur(imgOut, 15)
    imgOut = cv2.GaussianBlur(imgOut, (11, 21), 0)
    #return imgOut
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    imgOut = cv2.medianBlur(imgOut, 11)
    
    #imgOut = cv2.cvtColor(imgOut, cv2.COLOR_RGB2GRAY)
    imgOut = imgOut[:, :, 2]
    # return imgOut

    grad_x = cv2.Sobel(imgOut, cv2.CV_64F, 1, 0, ksize=31)
    grad_y = cv2.Sobel(imgOut, cv2.CV_64F, 0, 1, ksize=31)

    magnitude = cv2.magnitude(grad_x, grad_y)

    imgOut = cv2.normalize(magnitude, None, 220, 255, cv2.NORM_MINMAX).astype(np.uint8)
    imgOut = cv2.equalizeHist(imgOut)
    #return imgOut
    imgOut = cv2.equalizeHist(imgOut)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    imgOut = cv2.erode(imgOut, kernel, iterations=1)
    imgOut = cv2.medianBlur(imgOut, 11)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 9))
    imgOut = cv2.morphologyEx(imgOut, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    mask_players = utils.maskPlayers(image)
    mask_players = cv2.bitwise_not(mask_players)
    
    imgOut = cv2.bitwise_and(imgOut, mask_players)
    imgOut[imgOut > 0] = 255
    
    skeleton = np.zeros_like(imgOut)
    temp = np.zeros_like(imgOut)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        # Erosión
        eroded = cv2.erode(imgOut, kernel)
        # Apertura (erosión seguida de dilatación)
        opened = cv2.dilate(eroded, kernel)
        # Restar apertura de la erosión
        temp = cv2.subtract(imgOut, opened)
        # Acumular el resultado en el esqueleto
        skeleton = cv2.bitwise_or(skeleton, temp)
        # Actualizar la máscara
        imgOut = eroded.copy()

        # Si la máscara está completamente erosionada, detener
        if cv2.countNonZero(imgOut) == 0:
            break

    #return skeleton
    
    #return imgOut
    lines = cv2.HoughLinesP(skeleton, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=300)
    return lines