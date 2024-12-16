import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

def maskField(image):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_green = np.array([35, 20, 20])
    upper_green = np.array([85, 255, 255])
    
    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask_cleaned = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel, iterations=3)
    
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask_final = cv2.dilate(mask_cleaned, kernel_dilate, iterations=2)
    
    contours = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    maxContour = max(contours, key=cv2.contourArea)
    
    contourMask = np.zeros_like(mask_final)
    cv2.drawContours(contourMask, [maxContour], -1, 255, -1)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    contourMask = cv2.morphologyEx(contourMask, cv2.MORPH_CLOSE, kernel, iterations=5)
    
    return contourMask

def maskPlayers(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_green = np.array([35, 20, 20])
    upper_green = np.array([85, 255, 255])
    
    mask_field = maskField(image)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_players = cv2.bitwise_not(mask_green)
    mask_players = cv2.bitwise_and(mask_players, mask_field)
    #return mask_players
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 2))
    mask_players = cv2.morphologyEx(mask_players, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 15))
    mask_players = cv2.morphologyEx(mask_players, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_players = cv2.morphologyEx(mask_players, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    contours = cv2.findContours(mask_players, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    maxContour = max(contours, key=cv2.contourArea)
    print(maxContour)
    contourMask = np.zeros_like(mask_players)
    cv2.drawContours(contourMask, [maxContour], -1, 255, -1)
    
    mask_players = cv2.bitwise_and(mask_players, cv2.bitwise_not(contourMask))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_players = cv2.morphologyEx(mask_players, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_players = cv2.morphologyEx(mask_players, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    #return mask_players
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1), anchor=(0,0))
    mask_players = cv2.dilate(mask_players, kernel, iterations=1)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    mask_players = cv2.morphologyEx(mask_players, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    
    return mask_players

def drawPlayers(img, contours):
    imgOut = img.copy()
    
    for contour in contours:
        #if cv2.contourArea(contour) > 1000:
        #    continue
        
        x, y, w, h = cv2.boundingRect(contour)
        #if w < 500 and h < 400 and w <= 1.3 * h:
        cv2.rectangle(imgOut, (x, y - 10), (x + w, y + h + 10), (255, 0, 0), 2)
    
    return imgOut

def lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
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
            #filteredLines.append((x1, y1, x2, y2, 1, 1))
            #continue
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