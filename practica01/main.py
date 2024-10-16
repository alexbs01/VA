import functions as f
import cv2
import skimage as ski
import numpy as np
import utils

def main():
    # Blur
    kernel01 = np.array([[1, 1, 1], 
                        [1, 1, 1], 
                        [1, 1, 1]], dtype=np.float32) / 9.0
        
    # Detección de bordes
    kernel02 = np.array([[-1, -1, -1], 
                        [-1,  8, -1], 
                        [-1, -1, -1]])
    
    img = cv2.imread('imagenesPrueba/grid.png', cv2.IMREAD_GRAYSCALE)
    img = ski.util.img_as_float(img)
    nbins = 256
    
    #imgOut = f.adjustIntensity(img, [0.4, 0.60], [0., 1.])
    #imgOut = f.equalizeIntensity(img, nbins)
    #imgOut = f.filterImage(img, kernel02)
    #imgOut = f.gaussianFilter(img, 1)
    imgOut = f.medianFilter(img, 3)

    print(f.gaussKernel1D(1))
    utils.show_imgs_and_histogram(img, imgOut, nbins)


if __name__ == "__main__":
    main()