from rqst import *

def dado_imagen(frame):
    """
    Esta Función es la encargada de encontrar los dados para el video encontrando los "hijos con hierachy"
    """
    _, frame_bin = cv2.threshold(frame, thresh=85, maxval=255, type=cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    frame_bin = cv2.morphologyEx(frame_bin, cv2.MORPH_OPEN, kernel,iterations=3)
    # frame_bin = cv2.morphologyEx(frame_bin, cv2.MORPH_CLOSE, kernel)


    contours, _= cv2.findContours(frame_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    imagen_salida = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    imshow(frame_bin, title="Después de apertura (Opening)",blocking=True)
    dados = []

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        # if 300 < area < 2500 and h<52:
        dados.append(contour)
        
        margen_x = int(0.2 * w)  # Expandir 20% del ancho
        margen_y = int(0.2 * h)  # Expandir 20% de la altura

        # Coordenadas ajustadas
        x = max(x - margen_x, 0)  # No permitir valores negativos
        y = max(y - margen_y, 0)
        w = w + 2 * margen_x
        h = h + 2 * margen_y
        # Dibuja el bounding box en la imagen
        cv2.rectangle(imagen_salida, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostrar la imagen con los contornos internos y externos de los dados
    # imshow(imagen_salida, title='Contornos Filtrados Internos y Externos')
    return imagen_salida, dados

def detectar_movimiento(video_path, umbral_movimiento=5000, escala=0.5):
    """
    Detecta cuándo los objetos en un video se quedan quietos basándose en la diferencia entre frames consecutivos.
    También indica el número de píxeles donde se detecta movimiento.

    Parámetros:
    - video_path (str): Ruta del archivo de video.
    - umbral_movimiento (int): Valor que define el nivel de cambio para considerar movimiento.
    - escala (float): Escala para redimensionar el video (ej. 0.5 para reducir al 50% del tamaño original).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("No se pudo abrir el video.")
        return

    # Leer el primer frame
    ret, frame_actual = cap.read()
    if not ret:
        print("No se pudo leer el primer frame.")
        cap.release()
        return

    frame_actual = cv2.cvtColor(frame_actual, cv2.COLOR_BGR2GRAY)
    frame_actual = cv2.resize(frame_actual, None, fx=escala, fy=escala, interpolation=cv2.INTER_LINEAR)
    frame_number = 0

    while cap.isOpened():
        # Leer el siguiente frame
        ret, frame_siguiente = cap.read()
        if not ret:
            print("Fin del video.")
            break

        frame_siguiente = cv2.cvtColor(frame_siguiente, cv2.COLOR_BGR2GRAY)
        frame_siguiente = cv2.resize(frame_siguiente, None, fx=escala, fy=escala, interpolation=cv2.INTER_LINEAR)

        # Vemos si cambia el frame actual
        diferencia = cv2.absdiff(frame_actual, frame_siguiente)

        # Usamos un umbral para resaltar las diferencias
        _, diferencia_bin = cv2.threshold(diferencia, 30, 255, cv2.THRESH_BINARY)

        # Vemos cuantos pixeles se mueven para retener los que son 0
        num_pixeles_movimiento = np.sum(diferencia_bin > 0)
        
        if num_pixeles_movimiento == 0 and frame_number > 50:
            # imshow(frame_actual)
            return frame_actual, frame_number
        frame_number+=1
        
        frame_actual = frame_siguiente
        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("Detenido por el usuario.")
            break
    cap.release()
    cv2.destroyAllWindows()
for i in range(1,5):
    video_path = f"data/tirada_{i}.mp4"

    frame, numero_frame = detectar_movimiento(video_path, umbral_movimiento=0, escala=0.5)

    _, frame_bin = cv2.threshold(frame, thresh=80, maxval=255, type=cv2.THRESH_BINARY)

    imagen,dados = dado_imagen(frame)

    imshow(imagen,blocking=True)

frame_c, circulos = find_and_draw_circles(frame_bin,dp_ = 1.2, minD=7,minR=2,p1=1,p2=8,maxR=8)

# print(len(circulos))
# print(numero_frame)
