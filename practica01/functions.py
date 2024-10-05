import numpy as np

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
    
    # Paso 1: Obtener el histograma de la imagen utilizando tu función _histogram
    hist = _histogram(image, nBins)
    
    # Paso 2: Calcular la función de distribución acumulada (CDF)
    cdf = hist.cumsum()  # Sumatoria acumulada del histograma
    cdf_normalized = cdf / cdf[-1]  # Normalizamos la CDF para que vaya de 0 a 1
    
    # Paso 3: Generar los intervals para mapear los valores
    intervals = np.linspace(0, 1, nBins + 1)  # Crear los intervalos igual que en tu código
    
    # Paso 4: Usar la CDF para mapear los valores originales de la imagen a los nuevos
    image_equalized = np.interp(image.flatten(), intervals[:-1], cdf_normalized)
    
    # Paso 5: Reformar la imagen a su tamaño original
    outImage = image_equalized.reshape(inImage.shape)
    
    return outImage

def _histogram(inImage, nBins=256):
    """
    histogram
    
    Descripción: Calcula el histograma de una imagen.
    """
    
    image = inImage.flatten()  # Aplanar la imagen en una dimensión
    
    hist = np.zeros(nBins, dtype=int)  # Inicializar el histograma en ceros
    
    # Crear los intervalos entre 0 y 1
    intervals = np.linspace(0, 1, nBins + 1)
    
    # Llenar el histograma contando los píxeles que caen en cada intervalo
    for i in range(nBins):
        hist[i] = np.sum((image >= intervals[i]) & (image < intervals[i + 1]))
    
    # Asegurarse de que el último bin capture los valores igual a 1
    hist[-1] = np.sum(image == 1)
    
    return hist

