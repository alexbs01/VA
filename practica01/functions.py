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
    
    outImage = np.clip(outImage, 0., 1.)
    
    return outImage

def filterImage(inImage, kernel):
    """
    filterImage

    Descripción: Filtrar una imagen con un kernel de convolución.
        Con numpy se puede hacer una convolución de una matriz con otra matriz.
        Con los píxeles que quedan fuera de la imagen se asumen como 0.
    """
    
    outImage = np.copy(inImage)
    
    imagePadded = np.pad(outImage, 1, mode='constant')
    
    rows, columns = outImage.shape
    rowsKernel, columnsKernel = kernel.shape

    
    for i in range(rows):
        for j in range(columns):
            outImage[i, j] = np.sum(imagePadded[i:i + rowsKernel, j:j + columnsKernel] * kernel)
    
    outImage = np.clip(outImage, 0., 1.)
    
    return outImage


def gaussKernel1D(sigma):
    """    
    Parámetros
    ----------
    
    - sigma: Desviación estándar de la distribución normal.
    
    Return
    ----------
    - kernel: Vector 1xN con el kernel de la distribución normal, teniendo en cuenta que:
        - El centro x = 0 de la Gaussiana está en la posición ⌊N/2⌋ + 1.
        - N se calcula a partir de σ como N = 2⌈3σ⌉ + 1
    """

    N = 2 * np.ceil(3 * sigma) + 1
    x = np.arange(-N // 2, N // 2 + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel /= np.sum(kernel)
    
    return kernel