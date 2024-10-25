import numpy as np
import matplotlib.pyplot as plt

def histogram(inImage, nBins=256):
    """
    histogram
    
    Descripción: Calcula el histograma de una imagen.
    """
    
    image = inImage.flatten()
    
    hist = np.zeros(nBins, dtype=int)
    
    # Intervalos entre 0 y 1
    intervals = np.linspace(0, 1, nBins + 1)
    
    for i in range(nBins):
        hist[i] = np.sum((image >= intervals[i]) & (image < intervals[i + 1]))
    
    hist[-1] = np.sum(image == 1)
    
    return hist, intervals

def getRegion(image, kernel, i, j):
    rows_kernel, columns_kernel = kernel.shape
    
    imagePadded = np.pad(image, ((rows_kernel // 2, rows_kernel // 2), (columns_kernel // 2, columns_kernel // 2)), constant_values=0, mode='constant')
    
    return imagePadded[i:i + rows_kernel, j:j + columns_kernel]

def compareForErode(region, SE):    
    rows, columns = SE.shape
    
    for i in range(rows):
        for j in range(columns):
            if SE[i, j] == 1 and region[i, j] == 0:
                return False
    
    return True

def compareForDilate(region, SE):
    rows, columns = SE.shape
    
    for i in range(rows):
        for j in range(columns):
            if SE[i, j] == 1 and region[i, j] == 1:
                return True
    
    return False

def centerMatrix(matrix):
    """
    centerMatrix
    
    Descripción: Retorna el centro de una matriz
    """
    
    rows, columns = matrix.shape
    
    return [np.floor(rows // 2).astype(int), np.floor(columns // 2).astype(int)]

def operatorRoberts(image):
    rows, columns = image.shape
    regionSize = np.zeros([3, 3])
    
    gx = 0
    gy = 0
    
    for row in range(rows):
        for column in range(columns):
            region = getRegion(image, regionSize, row, column)
            
            gx += region[2, 2] - region[1, 1]
            gy += region[2, 1] - region[1, 2]
            
    
    print("gx:", gx)
    print("gy:", gy)
    gx /= rows * columns
    gy /= rows * columns
    print("gx:", gx)
    print("gy:", gy)
    return [gx, gy]

def operatorCentralDiff(image):
    pass

def operatorPrewitt(image):
    pass

def operatorSobel(image):
    pass

def show_imgs_and_histogram(img01, img02, nbins=256):
    # Crear una figura con 2x2 subplots: dos para las imágenes y dos para los histogramas
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Mostrar la primera imagen
    axs[0, 0].imshow(img01, cmap='gray')
    axs[0, 0].set_title('Imagen original')
    axs[0, 0].axis('off')  # Ocultar ejes

    # Mostrar la segunda imagen
    axs[0, 1].imshow(img02, cmap='gray')
    axs[0, 1].set_title('Imagen procesada')
    axs[0, 1].axis('off')  # Ocultar ejes

    # Calcular el histograma de la primera imagen
    histograma1 = histogram(img01, nbins)[0]

    # Graficar el histograma de la primera imagen
    axs[1, 0].plot(histograma1, color='black')
    axs[1, 0].set_title('Histograma de imagen original')
    axs[1, 0].set_xlim([0, nbins])  # Limitar el rango del eje x (niveles de gris)

    # Calcular el histograma de la segunda imagen
    histograma2 = histogram(img02, nbins)[0]

    # Graficar el histograma de la segunda imagen
    axs[1, 1].plot(histograma2, color='black')
    axs[1, 1].set_title('Histograma de imagen procesada')
    axs[1, 1].set_xlim([0, nbins])  # Limitar el rango del eje x (niveles de gris)

    # Ajustar los márgenes y mostrar las gráficas
    plt.tight_layout()
    plt.show()
