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
    
    img = cv2.imread('imagenesPrueba/eq0.png', cv2.IMREAD_GRAYSCALE)
    img = ski.util.img_as_float(img)

    #imgOut = f.adjustIntensity(img, [0.5, 0.51], [0., 1.])
    imgOut = f.equalizeIntensity(img, 256)
    #imgOut = f.filterImage(img, kernel02)
    
    print(f.gaussKernel1D(1))

    cv2.imshow("image", imgOut)
    #cv2.waitKey()


if __name__ == "__main__":
    main()