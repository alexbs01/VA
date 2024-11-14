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
                        [1, 1, 1]], dtype=np.float32)
        
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
    
    functions = {
        "intensity": 1,         
        "equalize": 2,          # eq0.png
        "filter": 3,            # circles.png
        "gaussian": 4,          # grid.png
        "median": 5,            # grid.png
        "erode": 6,             # morph.png
        "dilate": 7,            # morph.png
        "opening": 8,           # morph.png
        "closing": 9,           # morph.png
        "fill": 10,             # image.png
        "gradient": 11,         # circles1.png
        "LoG": 12,              # circles1.png
        "edgeCanny": 13         # circles1.png
    }
    
    img = cv2.imread('imagenesPrueba/image.png', cv2.IMREAD_GRAYSCALE)
    img = ski.util.img_as_float(img)
    
    inRange = [0.4, 0.60]
    outRange = [0., 1.]
    nbins = 256
    kernelToUse = kernel02
    sigma = 1.0
    filter_size = 3
    se = se01
    seeds = [[30, 30]]
    center = []
    operator = "sobel"
    tlow = 0.02
    thigh = 0.6
    
    function_to_execute = functions["fill"]
    
    match function_to_execute:
        case 1:
            imgOut = f.adjustIntensity(img, inRange=inRange, outRange=outRange)
        case 2:
            imgOut = f.equalizeIntensity(img, nBins=nbins)
        case 3:
            imgOut = f.filterImage(img, kernel=kernelToUse)
        case 4:
            imgOut = f.gaussianFilter(img, sigma=sigma)
        case 5:
            imgOut = f.medianFilter(img, filterSize=filter_size)
        case 6:
            imgOut = f.erode(img, se, center=center)
        case 7:
            imgOut = f.dilate(img, se, center=center)
        case 8:
            imgOut = f.opening(img, se, center=center)
        case 9:
            imgOut = f.closing(img, se, center=center)
        case 10:
            imgOut = f.fill(img, seeds=seeds, center=center)
        case 11:
            gradient = f.gradientImage(img, operator)
        case 12:
            imgOut = f.LoG(img, sigma=sigma)
        case 13:
            imgOut = f.edgeCanny(img, sigma=sigma, tlow=tlow, thigh=thigh)
        case _:
            print("Function not found")
    
    if imgOut is not None:
        utils.show_imgs_and_histogram(img, imgOut, nbins)
    else:
        print("Gradient:", gradient)
        utils.show_imgs_and_histogram(img, gradient[0] + gradient[1], nbins)


if __name__ == "__main__":
    main()