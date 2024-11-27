#Librerías
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
from PIL import Image 

# Función para mostrar imágenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    if title:
        plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)
        
def find_and_draw_circles(thresh_image, dp_ = 1.2, minD = 30, p1 = 50, p2 = 100, minR = 50, maxR = 500)-> tuple[np.array, list]:
    """
    #### Función para encontrar círculos en una imagen binarizada (thresh_image)
    #### y dibujarlos sobre una nueva imagen.
    ----------------------------------------------------------------------
    #### Parámetros:
        - thresh_image (numpy.ndarray): Imagen binarizada en la que se buscarán los círculos.
        - dp _ : valor de dp para HoughCircles.
        - minD: distancia mínima entre un centroide de un círculo y otro.
        - p1: parámetro 1 de HoughCircles.
        - p2: parámetro 2 de HoughCircles a mayor número, más exigente en la detección de círculos.
        - minR: radio mínimo de círculo.
        - maxR: radio máximo de círculo.

    ----------------------------------------------------------------------

    #### Retorna:
        - result_image (numpy.ndarray): Imagen con los círculos dibujados sobre la original.
        - circulos: circulos encontrados: (list[(x,y,r)])
    
    ----------------------------------------------------------------------

    #### Procedimiento:
        1. Encuentra los círculos en la imagen mediante HoughCircles.
        2. Dibuja los círculos en la imagen.
        3. Retorna la imagen con los círculos dibujados y una lista de dichos
            círculos.
    """
    # Aplicar la Transformada de Hough para detectar círculos
    # El primer parámetro es la imagen de entrada, debe ser en escala de grises o binarizada
    circles = cv2.HoughCircles(thresh_image, 
                               cv2.HOUGH_GRADIENT, dp=dp_, minDist=minD, 
                               param1=p1, param2=p2, minRadius=minR, maxRadius=maxR)
    
    # Si se encuentran círculos
    if circles is not None:
        # Convertir las coordenadas de los círculos a enteros
        circles = np.round(circles[0, :]).astype("int")
        
        # Crear una imagen copia de la original para dibujar los círculos
        result_image = cv2.cvtColor(thresh_image, cv2.COLOR_GRAY2BGR)  # Convertir a BGR para poder dibujar círculos
        

        # Dibujar los círculos encontrados
        for (x, y, r) in circles:
            area = np.pi * r **2
            print(area)
            # Dibujar el círculo exterior
            cv2.circle(result_image, (x, y), r, (0, 255, 0), 1)
            # Dibujar el centro del círculo
            cv2.circle(result_image, (x, y), 2, (0, 0, 255), 1)

            # Escribir el área del círculo en la imagen
            # cv2.putText(result_image, f'Area: {area:.2f}', (x - 40, y - r - 10),
            #     cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)
        
        return result_image,circles 
        # Mostrar la imagen con los círculos dibujados
        #imshow(result_image)
    
    else:
        return ("No se encontraron círculos en la imagen.")
    