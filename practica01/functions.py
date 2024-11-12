import numpy as np
import utils

####################################
# HISTOGRAMAS: MEJORA DE CONTRASTE #
####################################

def adjustIntensity(inImage, inRange=[], outRange=[0., 1.]):
    """
    Ajusta la intensidad de una imagen. De entrada recibe un intervalo, todo lo que esté por arriba o abajo saturará a los valores de los extremos.
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
    Expande donde hay mucha información y comprime donde hay poca.
    
    nBins: Número de intervalos en los que se divide el histograma.
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


################################
# FILTRADO ESPACIAL: SUAVIZADO #
################################

def filterImage(inImage, kernel):
    """
    Filtrar una imagen con un kernel de convolución.
        Con numpy se puede hacer una convolución de una matriz con otra matriz.
        Con los píxeles que quedan fuera de la imagen se asumen como 0.
    """
    rows, columns = inImage.shape
    
    outImage = np.zeros((rows, columns))
    
    for i in range(rows):
        for j in range(columns):
            region = utils.getRegion(inImage, kernel, i, j)

            outImage[i, j] = np.sum(region * kernel)

    
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

    N = int(2 * np.ceil(3 * sigma) + 1)
    x = np.arange(-N // 2, N // 2 + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel /= np.sum(kernel)
    
    return kernel

def gaussianFilter(inImage, sigma):
    """
    Nota. Como el filtro Gaussiano es lineal y separable podemos implementar este suavizado 
    simplemente convolucionando la imagen, primero, con un kernel Gaussiano unidimensional 1 x N 
    y, luego, convolucionando el resultado con el kernel transpuesto N x 1.
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


###########################
# OPERADORES MORFOLÓGICOS #
###########################

def erode(inImage, SE, center=[]):
    """
    Se recorre la matriz de la imagen y se compara con el elemento estructurante.
    Si todos los elementos del elemento estructurante son 1 y los elementos de la región
    de la imagen son 1, entonces se asigna 1 a la imagen de salida.
    
    Buscar la región de la imagen que coincida con el elemento estructurante.
    
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
            
            if utils.compareForErode(region, SE):
                outImage[i, j] = 1
    
    return outImage

def dilate(inImage, SE, center=[]):
    """
    Se recorre la matriz de la imagen y se compara con el elemento estructurante.
    Si al menos un elemento del elemento estructurante es 1 y los elementos de la región
    de la imagen son 1, entonces se asigna 1 a la imagen de salida.
    
    Agrandar los objetos de la imagen.
    
    - SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante.
    - center: Vector 1x2 con las coordenadas del centro de SE. Se asume que el [0 0] es
    la esquina superior izquierda. Si es un vector vacío (valor por defecto), el centro
    se calcula como (⌊P/2⌋ + 1, ⌊Q/2⌋ + 1).
    """
    if center == []:
        center = utils.centerMatrix(SE)
    
    outImage = np.zeros(inImage.shape)
    
    rows, columns = inImage.shape
    
    for i in range(rows):
        for j in range(columns):
            region = utils.getRegion(inImage, SE, i, j)
            region = np.round(region)
            
            if utils.compareForDilate(region, SE):
                outImage[i, j] = 1
    
    return outImage

def opening(inImage, SE, center=[]):
    """
    Erosionar y luego dilata usando el mismo elemento estructurante
    
    Buscar la región de la imagen que coincida con el elemento estructurante 
    filtrando de la imagen el resto de los objetos.
    
    - SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante.
    - center: Vector 1x2 con las coordenadas del centro de SE. Se asume que el [0 0] es
    la esquina superior izquierda. Si es un vector vacío (valor por defecto), el centro
    se calcula como (⌊P/2⌋ + 1, ⌊Q/2⌋ + 1).
    """
    outImage = erode(inImage, SE, center)
    
    outImage = dilate(outImage, SE, center)
    
    return outImage

def closing(inImage, SE, center=[]):
    """
    Primero dilatar y luego erosionar usando el mismo elemento estructurante
    
    Agrandar los objetos de la imagen y luego reducir el tamaño de los objetos
    Suele ser útil para cerrar pequeños huecos.
    
    - SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante.
    - center: Vector 1x2 con las coordenadas del centro de SE. Se asume que el [0 0] es
    la esquina superior izquierda. Si es un vector vacío (valor por defecto), el centro
    se calcula como (⌊P/2⌋ + 1, ⌊Q/2⌋ + 1).
    """
    
    outImage = dilate(inImage, SE, center)
    
    outImage = erode(outImage, SE, center)
    
    return outImage

def fill(inImage, seeds, SE=[], center=[]):
    """
    Rellenar un objeto de la imagen a partir de un conjunto de puntos semilla.
    
    seeds: Matriz Nx2 con N coordenadas (fila,columna) de los puntos semilla.
    SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante de conectividad.
        Si es un vector vacío se asume conectividad 4 (cruz 3 x 3).
    """
    if SE == []:
        SE = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]])

    if center == []:
        center = utils.centerMatrix(SE)
        
    outImage = np.copy(inImage)
    
    for seed in seeds:
        rowSeed, columnSeed = seed
        if inImage[rowSeed, columnSeed] == 0:
            outImage[rowSeed, columnSeed] = 1
        
        region = utils.getRegion(outImage, SE, rowSeed, columnSeed)
        region = np.round(region)
        
        for i in range(region.shape[0]):
            for j in range(region.shape[1]):
                if SE[i, j] == 1 and region[i, j] == 0:
                    outImage[rowSeed + i - center[0], columnSeed + j - center[1]] = 1
                    seeds.append([rowSeed + i - center[0], columnSeed + j - center[1]])
    
    return outImage


#######################
# DETECCIÓN DE BORDES #
#######################

def gradientImage(inImage, operator):
    """
    Implementar una función que permita obtener las componentes Gx y Gy del gradiente de una
    imagen, pudiendo elegir entre los operadores de Roberts, CentralDiff (Diferencias centrales
    de Prewitt/Sobel sin promedio: i.e. [-1, 0, 1] y transpuesto), Prewitt y Sobel
    
    operator: Permite seleccionar el operador utilizado mediante los valores: 'Roberts',
    'CentralDiff', 'Prewitt' o 'Sobel'.
    
    Return
    ----------
    [gx, gy]: Componentes Gx y Gy del gradiente.
    
    Usar atan2
    """
    operator = operator.lower()
    
    gradient = np.zeros(shape=[1, 2])
    
    match operator:
        case "roberts":
            gradient = utils.operatorRoberts(inImage)
        case "centraldiff":
            gradient = utils.operatorCentralDiff(inImage)
        case "prewitt":
            gradient = utils.operatorPrewitt(inImage)
        case "sobel":
            gradient = utils.operatorSobel(inImage)
        case _:
            raise "Operator debe de ser Roberts, CentralDiff, Prewitt o Sobel"
    
    return gradient

def LoG(inImage, sigma):
    """
    Implementa el filtro Laplaciano de Gauss (LoG).
    
    Parámetros
    ----------
    - inImage: Imagen de entrada en escala de grises.
    - sigma: Desviación estándar del filtro Gaussiano.
    
    Retorno
    -------
    - outImage: Imagen resultante después de aplicar el filtro LoG.
    """
    suavizada = gaussianFilter(inImage, sigma)
    
    laplaciano_kernel = np.array([[1, 1, 1], 
                                [1, -8, 1], 
                                [1, 1, 1]])
    
    outImage = filterImage(suavizada, laplaciano_kernel)
    
    return outImage

def edgeCanny(inImage, sigma, tlow, thigh):
    """
    Implementar el detector de bordes de Canny
    
    sigma: Parámetro sigma del filtro Gaussiano.
    tlow, thigh: Umbrales de histéresis bajo y alto, respectivamente.
    1. Guassiana
    2. Gradiente
    3. Supresión no máxima
    4. Histéresis
    """
    assert tlow <= thigh, "tlow must be less than thigh"
    
    image = np.copy(inImage)
    
    image = gaussianFilter(image, sigma)
    
    jx, jy = gradientImage(image, "Sobel")
    
    gradientDirection = np.atan2(jy, jx)
    gradientMagnitude = np.sqrt(jx**2 + jy**2)
    
    beforeHysteresis = utils.NMS(gradientDirection, gradientMagnitude)
    
    afterHysteresis = utils.hysteresis(beforeHysteresis, gradientDirection, tlow, thigh)
        
    return afterHysteresis