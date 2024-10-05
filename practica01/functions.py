import numpy as np
import utils

def adjustIntensity(inImage, inRange=[], outRange=[0., 1.]):
    """
    adjustIntensity

    Descripción: Ajusta la intensidad de una imagen. De entrada recibe un intervalo, todo lo que esté por arriba o abajo saturará a los valores de los extremos.
        y = mx + b
        m: (Mo - mo) / (Mi - mi)
        b: Mo - m * Mi
    """
    
    assert len(inRange) == 2, "inRange must be a list of two elements"
    assert len(outRange) == 2, "outRange must be a list of two elements"
    
    image = np.copy(inImage)
    
    inMin, inMax = inRange
    outMin, outMax = outRange
    
    m = (outMax - outMin) / (inMax - inMin)
    b = outMax - m * inMax
    
    outImage = image * m + b
    outImage = np.clip(outImage, outMin, outMax)

    return outImage
    


def equalizeIntensity(inImage, nBins=256):
    """
    equalizeIntensity
    
    Descripción: Expande donde hay mucha información y comprime donde hay poca.
    """
    
    # Hacer una copia de la imagen para no modificar la original
    image = np.copy(inImage)
    
    hist, intervals = utils.histogram(image, nBins)
    
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]
    
    image_equalized = np.interp(image.flatten(), intervals[:-1], cdf_normalized)
    
    outImage = image_equalized.reshape(inImage.shape)
    
    return outImage

def filterImage(inImage, kernel):
    """
    filterImage

    Descripción: Filtrar una imagen con un kernel de convolución.
        Con numpy se puede hacer una convolución de una matriz con otra matriz.
        Con los píxeles que quedan fuera de la imagen se asumen como 0.
    """
    
    # Hacer una copia de la imagen para no modificar la original
    imageOut = np.copy(inImage)
    
    imagePadded = np.pad(imageOut, 1, mode='constant')
    
    rows, columns = imageOut.shape
    rowsKernel, columnsKernel = kernel.shape
    
    for i in range(rows):
        for j in range(columns):
            imageOut[i, j] = np.sum(imagePadded[i:i + rowsKernel, j:j + columnsKernel] * kernel)
    
    
    return imageOut
    