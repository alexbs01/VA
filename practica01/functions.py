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
    rows, columns = inImage.shape
    
    result = np.zeros((rows, columns))
    
    for i in range(rows):
        for j in range(columns):
            region = utils.getRegion(inImage, kernel, i, j)
            result[i, j] = np.sum(region * kernel)
    
    outImage = np.clip(result, 0., 1.)
    
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

def gaussianFilter(inImage, sigma):
    """
    Nota. Como el filtro Gaussiano es lineal y separable podemos implementar este suavi-
    zado simplemente convolucionando la imagen, primero, con un kernel Gaussiano unidi-
    mensional 1 x N y, luego, convolucionando el resultado con el kernel transpuesto N x 1.
    """
    
    kernel = gaussKernel1D(sigma)
    kernel = kernel.reshape(1, -1)
    
    outImage = filterImage(inImage, kernel)
    outImage = filterImage(outImage, np.transpose(kernel))
    
    return outImage

def medianFilter(inImage, filterSize):
    """
    Parámetros
    -----
    
    - filterSize: Valor entero N indicando que el tamaño de ventana es de NxN. La posición
    central de la ventana es (⌊N/2⌋ + 1, ⌊N/2⌋ + 1).
    """
    
    rows, columns = inImage.shape
    
    result = np.zeros((rows, columns))
    
    for i in range(rows):
        for j in range(columns):
            region = utils.getRegion(inImage, np.ones((filterSize, filterSize)), i, j)
            result[i, j] = np.median(region)
    
    return result

def erode(inImage, SE, center=[]):
    """
    - SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante.
    - center: Vector 1x2 con las coordenadas del centro de SE. Se asume que el [0 0] es
    la esquina superior izquierda. Si es un vector vacío (valor por defecto), el centro
    se calcula como (⌊P/2⌋ + 1, ⌊Q/2⌋ + 1).
    """
    if center == []:
        center = utils.centerMatrix(SE)
    
    outImage = np.zeros(inImage.shape)
    
    rows, columns = inImage.shape
    rowsSE, columnsSE = SE.shape
    
    for i in range(center[0], rows - rowsSE + 1):
        for j in range(center[1], columns - columnsSE + 1):
            region = utils.getRegion(inImage, SE, i, j)
            region = np.round(region)
            
            if utils.compareWithSE(region, SE):
                outImage[i + center[0], j + center[1]] = 1
    
    return outImage

def dilate(inImage, SE, center=[]):
    """
    - SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante.
    - center: Vector 1x2 con las coordenadas del centro de SE. Se asume que el [0 0] es
    la esquina superior izquierda. Si es un vector vacío (valor por defecto), el centro
    se calcula como (⌊P/2⌋ + 1, ⌊Q/2⌋ + 1).
    """
    pass

def opening(inImage, SE, center=[]):
    """
    - SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante.
    - center: Vector 1x2 con las coordenadas del centro de SE. Se asume que el [0 0] es
    la esquina superior izquierda. Si es un vector vacío (valor por defecto), el centro
    se calcula como (⌊P/2⌋ + 1, ⌊Q/2⌋ + 1).
    """
    pass

def closing(inImage, SE, center=[]):
    """
    - SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante.
    - center: Vector 1x2 con las coordenadas del centro de SE. Se asume que el [0 0] es
    la esquina superior izquierda. Si es un vector vacío (valor por defecto), el centro
    se calcula como (⌊P/2⌋ + 1, ⌊Q/2⌋ + 1).
    """
    pass
