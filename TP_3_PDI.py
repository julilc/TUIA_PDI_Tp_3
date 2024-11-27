# from rqst import *

# def leer_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS)) 
#     n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print(n_frames)
#     out = cv2.VideoWriter(f'salida_{video_path}', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
#     frame_number = 0
#     while (cap.isOpened()): # Verifica si el video se abrió correctamente.

#         ret, frame = cap.read() # 'ret' indica si la lectura fue exitosa (True/False) y 'frame' contiene el contenido del frame si la lectura fue exitosa.
#         # imshow(frame)
#         print(frame)
#         if ret == True:  

#             frame = cv2.resize(frame, dsize=(int(width/3), int(height/3))) # Redimensiona el frame capturado.

#             cv2.imshow('Frame', frame) # Muestra el frame redimensionado.

#             # cv2.imwrite(os.path.join("frames", f"frame_{frame_number}.jpg"), frame) # Guarda el frame en el archivo './frames/frame_{frame_number}.jpg'.

#             frame_number += 1
#             if cv2.waitKey(25) & 0xFF == ord('q'): # Espera 25 milisegundos a que se presione una tecla. Si se presiona 'q' se rompe el bucle y se cierra la ventana.
#                 break
#         else:  
#             break  

#     cap.release() # Libera el objeto 'cap', cerrando el archivo.
#     cv2.destroyAllWindows()
# # for i in range(1,5):
# #     leer_video(f'data/tirada_{i}.mp4')
# leer_video('data/tirada_1.mp4')
# cv2.HoughCircles(thresh_image, cv2.HOUGH_GRADIENT, dp=dp_, minDist=minD, 
#                 param1=p1, param2=p2, minRadius=minR, maxRadius=maxR)

from rqst import *

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

    # Verificar si se abrió el video correctamente
    if not cap.isOpened():
        print("No se pudo abrir el video.")
        return

    # Leer el primer frame
    ret, frame_actual = cap.read()
    if not ret:
        print("No se pudo leer el primer frame.")
        cap.release()
        return

    # Convertir a escala de grises y redimensionar
    frame_actual = cv2.cvtColor(frame_actual, cv2.COLOR_BGR2GRAY)
    frame_actual = cv2.resize(frame_actual, None, fx=escala, fy=escala, interpolation=cv2.INTER_LINEAR)
    frame_number = 0
    while cap.isOpened():
        # Leer el siguiente frame
        ret, frame_siguiente = cap.read()
        if not ret:
            print("Fin del video.")
            break

        # Convertir a escala de grises y redimensionar
        frame_siguiente = cv2.cvtColor(frame_siguiente, cv2.COLOR_BGR2GRAY)
        frame_siguiente = cv2.resize(frame_siguiente, None, fx=escala, fy=escala, interpolation=cv2.INTER_LINEAR)

        # Calcular la diferencia absoluta entre el frame actual y el siguiente
        diferencia = cv2.absdiff(frame_actual, frame_siguiente)

        # Aplicar un umbral para resaltar las diferencias significativas
        _, diferencia_bin = cv2.threshold(diferencia, 30, 255, cv2.THRESH_BINARY)

        # Contar el número de píxeles con movimiento (valores blancos en la diferencia binarizada)
        num_pixeles_movimiento = np.sum(diferencia_bin > 0)
        
        # Verificar si el movimiento está por debajo del umbral
        if num_pixeles_movimiento ==0 and frame_number > 50:
            # imshow(frame_actual)
            return frame_actual, frame_number
        frame_number+=1
        
        # Actualizar el frame actual
        frame_actual = frame_siguiente

        # Salir al presionar 'q'
        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("Detenido por el usuario.")
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

# Llamar a la función con la ruta del video
video_path = "data/tirada_4.mp4"
frame, numero_frame = detectar_movimiento(video_path, umbral_movimiento=5000, escala=0.5)
_, frame_bin = cv2.threshold(frame, thresh=80, maxval=255, type=cv2.THRESH_BINARY)
frame_c, circulos = find_and_draw_circles(frame_bin,minD=7,minR=2,p1=1,p2=8,dp_=1,maxR=8)
print(len(circulos))
print(numero_frame)
imshow(frame_c, blocking=True)
# cv2.waitKey(0)
# cv2.destroyAllWindows()