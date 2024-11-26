import matplotlib.pyplot as plt
import cv2
import numpy as np

def maskField(image):
    img = np.copy(image)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    imgRed = img[:, :, 0]
    imgGreen = img[:, :, 1]
    imgBlue = img[:, :, 2]
    
    threshold_green = 10
    mask_green = (imgGreen > imgRed + threshold_green) & (imgGreen > imgBlue + threshold_green)
    mask_green = mask_green.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5), anchor=(4, 0))
    mask_green = cv2.erode(mask_green, kernel=kernel, iterations=7)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_filled = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=7)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3), anchor=(15, 0))
    mask_aumented = cv2.dilate(mask_filled, kernel, iterations=5)
    
    return mask_aumented

def drawPlayers(img, contours):
    imgOut = img.copy()
    height, width = img.shape[:2]
    print(height*0.1, width*0.1)
    print("###")
    
    for contour in contours:
        # Calcular el área del contorno
        area = cv2.contourArea(contour)
        
        if 20*20 < area < 500*500:
            x, y, w, h = cv2.boundingRect(contour)
            print(w, h)
            if w < h * 1.3 and 20 < w < 500 and 20 < h < 500:
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