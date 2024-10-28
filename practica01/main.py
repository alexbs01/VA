import functions as f
import cv2
import skimage as ski
import numpy as np
import utils

def main():
    
    imgOut = None
    
    # Blur
    kernel01 = np.array([[1, 1, 1], 
                        [1, 1, 1], 
                        [1, 1, 1]], dtype=np.float32) / 9.0
        
    # Detección de bordes
    kernel02 = np.array([[-1, -1, -1], 
                        [-1,  8, -1], 
                        [-1, -1, -1]])
    
    se01 = np.array([[1, 1, 1]])
    
    se02 = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    
    se03 = np.array([[0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]])
    
    se04 = np.array([[1, 1]])
    
    
    img = cv2.imread('imagenesPrueba/circles1.png', cv2.IMREAD_GRAYSCALE)
    img = ski.util.img_as_float(img)
    nbins = 256
    
    #imgOut = f.adjustIntensity(img, [0.4, 0.60], [0., 1.])
    #imgOut = f.equalizeIntensity(img, nbins)               # eq0.png
    #imgOut = f.filterImage(img, kernel02)
    #imgOut = f.gaussianFilter(img, 1)                      # grid.png
    #imgOut = f.medianFilter(img, 3)                        # grid.png
    #imgOut = f.erode(img, se02)                            # morph.png
    #imgOut = f.dilate(img, se02)                           # morph.png
    #imgOut = f.opening(img, se03)                          # morph.png
    #imgOut = f.closing(img, se01)                          # morph.png
    #imgOut = f.fill(img, [[30, 30]])                       # image.png
    gradient = f.gradientImage(img, "roberts")
    
    #imgOut = f.LoG(img, 1)
    
    if imgOut is not None:
        utils.show_imgs_and_histogram(img, imgOut, nbins)
    else:
        print("Gradient:", gradient)
        utils.show_imgs_and_histogram(img, gradient[0] + gradient[1], nbins)


if __name__ == "__main__":
    main()