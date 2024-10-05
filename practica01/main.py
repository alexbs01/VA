import functions as f
import cv2
import skimage as ski
import numpy as np

def main():
    
    # Blur
    kernel01 = np.array([[1, 1, 1], 
                        [1, 1, 1], 
                        [1, 1, 1]], dtype=np.float32) / 9.0
        
    # Detección de bordes
    kernel02 = np.array([[-1, -1, -1], 
                        [-1,  8, -1], 
                        [-1, -1, -1]])
    
    img = cv2.imread('imagenesPrueba/mpp.png', cv2.IMREAD_GRAYSCALE)
    img = ski.util.img_as_float(img)

    print(img)
    print(img.shape)
    #imgOut = f.adjustIntensity(img, [0.4, 0.6], [0., 1.])
    imgOut = f.equalizeIntensity(img, 64)
    #imgOut = f.filterImage(img, kernel02)
    imgOut = f.adjustIntensity(imgOut, [0.1, 0.9], [0., 1])

    print(min(imgOut.flatten()), max(imgOut.flatten()))

    cv2.imshow("image", imgOut)
    cv2.waitKey()

if __name__ == "__main__":
    main()