# Detección de jugadores y líneas de céped

## Objetivos

Desarrollar un método computacional que permita:  

1. Detectar el área ocupada por el césped.
2. Detectar los objetos que hay sobre el terreno de juego: jugadores, árbitro y balón.
3. Localizar las líneas de siega del césped.

## Detección del área ocupada por el césped

### Metodología usada

1. **Conversión al modelo de color HSV**:  
   La imagen original se convierte al modelo de color HSV, ya que este facilita la segmentación basada en colores. Se definen dos rangos de color que abarcan tonos de verde típicos del césped.  

2. **Generiación de la máscara binaria**:  
   Se utilizan los rangos definidos para generar una máscar binaria donde los píxeles que corresponden al césped se establecen a 255, y el resto a 0.  

3. **Limpieza de máscara**:  
   - Se eliminan pequeñas partículas de ruido que no pertenecen al césped, como partes del público o zonas fuera del campo.  
   - Se rellenan los agujeros presentes en el césped, que suelen generarse por sombras o interrupciones causadas por jugadores y árbitros.  

## Detectar los objetos sobre el terreno de juego

### Metodología usada

1. **Aislar objetos no verdes**:  
   Partiendo de la máscar del césped obtenida, se genera una nueva máscar que retorna todo lo que no es verde que está dentro del campo. Estos objetos son jugadores, árbitros y el balón.

2. **Invertir y refinar la máscara**:  
   - Se invierten los colores para que la máscara tenga a los objetos del campo en blanco.  
   - Se aplican operaciones morforlógicas para eliminar ruido y agrandar las regiones detectadas.  

Cada área de la máscara es un objeto dentro del campo, por lo que se marcará.  

## Localizar las líneas de siega del césped

### Metodología usada

#### Detección de las líneas

1. **Preparación de la imagen**:  
   - Se limpia la imagen original para mejorar el contraste de las líneas de siega.  
   - Se utiliza la capa azul de la imagen, ya que en este canal suele haber mayor contraste entre las líneas y el césped.

2. **Detección de bordes**:  
   - Se aplica un detector de bordes basado en el gradiente (por ejemplo, Sobel) para resaltar las líneas del césped.

3. **Eliminación de interferencias**:  
   - Se resta la máscara de los jugadores y otros objetos detectados previamente, ya que estos generan interferencias que pueden afectar la detección de las líneas.

4. **Obtención del esqueleto**:
   - Se reduce el grosor de las líneas detectadas a un único píxel mediante la técnica del esqueleto, lo que facilita la detección precisa de líneas.

5. **Transformada de Hough**:  
   - Se aplica la Transformada de Hough para detectar las líneas rectas presentes en la imagen esquelética.

#### Filtrado y dibujo de las líneas

1. **Descartar líneas horizontales**:  
   - Las líneas horizontales no son relevantes para el análisis y se eliminan.

2. **Filtrado por intersecciones**:  
   - Se eliminan las líneas que se cruzan con al menos otra línea detectada, para evitar confusiones.

3. **Selección de la línea más representativa**:  
   - Si varias líneas están muy próximas entre sí, solo se conserva la más larga, ya que representa mejor la línea de siega.

4. **Dibujado final**:
   - Las líneas finales, que representan las líneas de siega más significativas, se dibujan sobre la imagen para facilitar su visualización.

### Metodologías descartadas

Durante el desarrollo del trabajo, se exploraron diversas alternativas que fueron descartadas debido a sus limitaciones:  

1. **Cálculo de la normal a las líneas de siega**:  
   - Se consideró calcular la normal a las líneas de siega para determinar la periodicidad entre ellas.
   - Sin embargo, la perspectiva del campo genera líneas no paralelas, lo que dificulta este enfoque.

2. **Punto de fuga**:  
   - Las líneas de siega convergen en un punto de fuga
   - Fue descartado debido a la complejidad de calcular un punto de fuga óptimo en imágenes con ruido o distorsión.

3. **Reducción de la gama de colores**
   - Se intentó reducir la gama de colores para distinguir las diferentes alturas del césped.
   - Este enfoque no funcionó en presencia de sombras o cuando los colores del césped eran demasiado similares.