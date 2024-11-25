import matplotlib.pyplot as plt
import cv2

def drawPlayers(img, contours):
    imgOut = img.copy()
    height, width = img.shape[:2]
    print(height*0.1, width*0.1)
    print("###")
    
    for contour in contours:
        # Calcular el área del contorno
        area = cv2.contourArea(contour)
        
        if 30*30 < area < 400*400:  # Umbral de área mínimo y máximo para rectángulos pequeños
            x, y, w, h = cv2.boundingRect(contour)  # Obtener el rectángulo delimitador
            print(w, h)
            if height * 0.05 < h < height * 0.3 and width * 0.03 < w < width * 0.2 and w < h * 1.2:
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