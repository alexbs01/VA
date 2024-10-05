"""
filterImage

Descripción: Filtrar una imagen con un kernel de convolución.
    Con numpy se puede hacer una convolución de una matriz con otra matriz.
    Con los píxeles que quedan fuera de la imagen se asumen como 0.
"""
import functions as f
import cv2
import skimage as ski

def main():
    
    img = cv2.imread('imagenesPrueba/eq0.png', cv2.IMREAD_GRAYSCALE)
    img = ski.util.img_as_float(img)

    print(img)
    print(img.shape)
    #imgOut = f.adjustIntensity(img, [0.4, 0.6], [0., 1.])
    imgOut = f.equalizeIntensity(img, 1000)

    print(min(imgOut.flatten()), max(imgOut.flatten()))

    cv2.imshow("image", imgOut)
    cv2.waitKey()

if __name__ == "__main__":
    main()