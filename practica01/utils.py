import numpy as np

def histogram(inImage, nBins=256):
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
    
    return hist, intervals

def centerMatrix(matrix):
    """
    centerMatrix
    
    Descripción: Retorna el centro de una matriz
    """
    
    rows, columns = matrix.shape
        
    return np.floor(rows / 2).astype(int) + 1, np.floor(columns / 2).astype(int) + 1
