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

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    imgOut = cv2.erode(imgOut, kernel, iterations=1)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    imgOut = cv2.morphologyEx(imgOut, cv2.MORPH_CLOSE, kernel, iterations=7)

    imgOut = cv2.cvtColor(imgOut, cv2.COLOR_RGB2GRAY)
    
    grad_x = cv2.Sobel(imgOut, cv2.CV_64F, 1, 0, ksize=15)
    grad_y = cv2.Sobel(imgOut, cv2.CV_64F, 0, 1, ksize=11)

    magnitude = cv2.magnitude(grad_x, grad_y)

    imgOut = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    imgOut = cv2.equalizeHist(imgOut)
    imgOut = cv2.medianBlur(imgOut, 11)

    for _ in range(5):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        imgOut = cv2.morphologyEx(imgOut, cv2.MORPH_CLOSE, kernel, iterations=1)

        imgOut = cv2.medianBlur(imgOut, 3)
        imgOut = cv2.morphologyEx(imgOut, cv2.MORPH_OPEN, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    imgOut = cv2.erode(imgOut, kernel, iterations=1)
    imgOut = cv2.medianBlur(imgOut, 3)
    

    lines = cv2.HoughLinesP(imgOut, 1, np.pi / 180, 200, minLineLength=100, maxLineGap=300)
    
    return lines

def prueba(image):
    imgOut = np.copy(image)

    imgOut = cv2.medianBlur(imgOut, 11)
    imgOut = cv2.GaussianBlur(imgOut, (5, 11), 0)
    #return imgOut
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    imgOut = cv2.medianBlur(imgOut, 11)
    imgOut = cv2.cvtColor(imgOut, cv2.COLOR_RGB2GRAY)
    
    grad_x = cv2.Sobel(imgOut, cv2.CV_64F, 1, 0, ksize=15)
    grad_y = cv2.Sobel(imgOut, cv2.CV_64F, 0, 1, ksize=11)

    magnitude = cv2.magnitude(grad_x, grad_y)

    imgOut = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    imgOut = cv2.equalizeHist(imgOut)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    imgOut = cv2.erode(imgOut, kernel, iterations=1)
    imgOut = cv2.medianBlur(imgOut, 11)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 9))
    imgOut = cv2.morphologyEx(imgOut, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    mask_players = utils.maskPlayers(image)
    mask_players = cv2.bitwise_not(mask_players)
    
    imgOut = cv2.bitwise_and(imgOut, mask_players)
    
    #return imgOut
    lines = cv2.HoughLinesP(imgOut, 5, np.pi / 180, 1000, minLineLength=100, maxLineGap=1000)
    
    return lines