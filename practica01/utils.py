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
    maskX = np.array([  [-1, 0], 
                        [0, 1]])
    maskY = np.array([  [0, -1],
                        [1, 0]])
    
    gx = np.zeros([rows, columns])
    gy = np.zeros([rows, columns])
    
    for row in range(rows):
        for column in range(columns):
            region = getRegion(image, maskX, row, column)
            
            gx[row, column] = np.sum(region * maskX)
            gy[row, column] = np.sum(region * maskY)
    
    return [gx, gy]

def operatorCentralDiff(image):
    rows, columns = image.shape
    maskX = np.array([  [-1, 0, 1]])
    maskY = np.transpose(maskX)
    
    gx = np.zeros([rows, columns])
    gy = np.zeros([rows, columns])
    
    for row in range(rows):
        for column in range(columns):
            region = getRegion(image, maskX, row, column)
            
            gx[row, column] = np.sum(region * maskX)
            gy[row, column] = np.sum(region * maskY)
    
    return [gx, gy]

def operatorPrewitt(image):
    rows, columns = image.shape
    maskX = np.array([  [-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]])
    maskY = np.array([  [-1, -1, -1],
                        [0, 0, 0],
                        [1, 1, 1]])
    
    gx = np.zeros([rows, columns])
    gy = np.zeros([rows, columns])
    
    for row in range(rows):
        for column in range(columns):
            region = getRegion(image, maskX, row, column)
            
            gx[row, column] = np.sum(region * maskX)
            gy[row, column] = np.sum(region * maskY)
    
    return [gx, gy]

def operatorSobel(image):
    rows, columns = image.shape
    maskX = np.array([  [-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    maskY = np.array([  [-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    
    gx = np.zeros([rows, columns])
    gy = np.zeros([rows, columns])
    
    for row in range(rows):
        for column in range(columns):
            region = getRegion(image, maskX, row, column)
            
            gx[row, column] = np.sum(region * maskX)
            gy[row, column] = np.sum(region * maskY)
    
    return [gx, gy]

def getGradientDirection(elementGradientDirection):
    angle = (2 * np.pi) / 16
    
    if elementGradientDirection <= angle and elementGradientDirection > -angle or \
        elementGradientDirection <= 9 * angle and elementGradientDirection > 7 * angle:
        return "horizontal"
    
    elif elementGradientDirection <= 3 * angle and elementGradientDirection > angle or \
        elementGradientDirection <= 11 * angle and elementGradientDirection > 9 * angle:
        return "first-quadrant"
    
    elif elementGradientDirection <= 5 * angle and elementGradientDirection > 3 * angle or \
        elementGradientDirection <= 13 * angle and elementGradientDirection > 11 * angle:
        return "vertical"
    
    elif elementGradientDirection <= 7 * angle and elementGradientDirection > 5 * angle or \
        elementGradientDirection <= 15 * angle and elementGradientDirection > 13 * angle:
        return "second-quadrant"
    
    else:
        return "horizontal"

def NMS(gradientDirection, gradientMagnitude):
    rows, cols = gradientMagnitude.shape
    print("NMS", rows, cols)
    
    for i in range(rows):
        for j in range(cols):
            elementGradientDirection = getGradientDirection(gradientDirection[i, j])
            
            if elementGradientDirection == "horizontal":
                if j + 1 < cols and gradientMagnitude[i, j] < gradientMagnitude[i, j + 1] or \
                    j - 1 >= 0 and gradientMagnitude[i, j] < gradientMagnitude[i, j - 1]:
                    gradientMagnitude[i, j] = 0
                    print("H", end="")
                
            elif elementGradientDirection == "first-quadrant":
                if i - 1 >= 0 and j + 1 < cols and gradientMagnitude[i, j] < gradientMagnitude[i - 1, j + 1] or \
                    i + 1 < rows and j - 1 >= 0 and gradientMagnitude[i, j] < gradientMagnitude[i + 1, j - 1]:
                    gradientMagnitude[i, j] = 0
                    print("1", end="")
                
            elif elementGradientDirection == "vertical":
                if i - 1 >= 0 and gradientMagnitude[i, j] < gradientMagnitude[i - 1, j] or \
                    i + 1 < rows and gradientMagnitude[i, j] < gradientMagnitude[i + 1, j]:
                    gradientMagnitude[i, j] = 0
                    print("V", end="")
                
            elif elementGradientDirection == "second-quadrant":
                if i - 1 >= 0 and j - 1 >= 0 and gradientMagnitude[i, j] < gradientMagnitude[i - 1, j - 1] or \
                    i + 1 < rows and j + 1 < cols and gradientMagnitude[i, j] < gradientMagnitude[i + 1, j + 1]:
                    gradientMagnitude[i, j] = 0
                    print("2", end="")
            
        print()
    
    return gradientMagnitude

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
