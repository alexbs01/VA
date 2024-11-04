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
    #imgOut = f.gaussianFilter(img, 0.1)                      # grid.png
    #imgOut = f.medianFilter(img, 3)                        # grid.png
    #imgOut = f.erode(img, se02)                            # morph.png
    #imgOut = f.dilate(img, se02)               rcles.png', cv2.IMREAD_GRAYSCALE)

    
    #imgOut = f.adjustIntensity(img, [0.4, 0.60], [0., 1.])
    #imgOut = f.equalizeIntensity(img, nbins)               # eq0.png
    #imgOut = f.filterImage(img, kernel02)            # morph.png
    #imgOut = f.opening(img, se03)                          # morph.png
    #imgOut = f.closing(img, se01)                          # morph.png
    #imgOut = f.fill(img, [[30, 30]])                       # image.png
    #gradient = f.gradientImage(img, "sobel")
    #imgOut = f.LoG(img, 1)
    imgOut = f.edgeCanny(img, 0.001, 0.5, 0.5)
    
    
    
    if imgOut is not None:
        utils.show_imgs_and_histogram(img, imgOut, nbins)
    else:
        print("Gradient:", gradient)
        utils.show_imgs_and_histogram(img, gradient[0] + gradient[1], nbins)
        for row in gradient[0]:
            for col in row:
                print(np.round(col, decimals=2), end="\t")
            print()
        print("AAA")
        for row in gradient[1]:
            for col in row:
                print(np.round(col, decimals=2), end="\t")
            print()
            
        print("BBB")
        for row in gradient[0] + gradient[1]:
            for col in row:
                print(np.round(col, decimals=2), end="\t")
            print()


if __name__ == "__main__":
    main()
    
    """
    La función atan2 calcula el ángulo en coordenadas polares a partir de las componentes Y (gradiente vertical) y X (gradiente horizontal) de una imagen.
    Cuando se aplica sobre los gradientes de una imagen, el resultado es un mapa de orientaciones que representa la dirección de los bordes o cambios de intensidad.
¿Cómo funciona?

    atan2(Y, X) devuelve el ángulo en radianes, tomando en cuenta los signos de ambos gradientes, lo que permite determinar el cuadrante correcto del ángulo.
    El ángulo resultante está comprendido entre -π y π, representando la dirección de los bordes respecto al eje X.

Aplicación en imágenes:

En procesamiento de imágenes, atan2 se utiliza para obtener la dirección del gradiente, que es útil en la detección de bordes y en algoritmos de análisis de 
contornos, como el operador de Canny o en la extracción de características de bordes en la imagen.

Por ejemplo:

    Si X y Y son las derivadas de la imagen en los ejes horizontal y vertical, atan2(Y, X) devuelve el ángulo que indica la dirección en la que la intensidad de 
    la imagen está cambiando más rápidamente.

Esto permite interpretar los bordes y su orientación, esencial para técnicas avanzadas de análisis de imágenes.
    """