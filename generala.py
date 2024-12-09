from rqst import *

def green_sample(frame:np.array)-> tuple[np.array, np.array,np.array]:
    '''
    Toma un frame y realiza una muesta de color verde a partir de la
    cual devuelve límites inferiores y superiores de rango de dicho color
    que sirve para hacer máscaras.  
    ----------------------------------------------------------------
    Parámetros:
    ----------------------------------------------------------------
        -frame: frame del cual se obtiene la muestra.
    
    -----------------------------------------------------------------
    Retorna:
    -----------------------------------------------------------------
        -upper_limmit: límite superior del color verde.
        -lower_limit: límite inferior del color verde.

    -----------------------------------------------------------------
    Procedimiento:
        1. Recorta una porción de la imagen donde está el fondo verde.
        2. Obtiene una muestra de color en RGB.
        3. Convierte dicha muestra a HSV.
        4. Define un delta.
        5. Con dicho delta hace un rango de color sobre la muestra HSV.
            definiendo lower_limit (límite inferior) y upper_limit
            (límite superior).
    -----------------------------------------------------------------
    
    '''
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bg_crop_RGB = frame_rgb[100:200, 100:200, :]
    img_pixels = bg_crop_RGB.reshape(-1,3)
    colours, counts = np.unique(img_pixels, axis = 0, return_counts= True)
    n_colours = colours.shape[0]
    n_colours
    colours[0]

    bg_color_rgb = colours[4]
    bg_color_rgb = np.reshape(bg_color_rgb, (1,1,-1))
    bg_color_hsv = cv2.cvtColor(bg_color_rgb, cv2.COLOR_RGB2HSV)
    DELTA_BG = np.array([10.0, 90.0, 90.0])
    #Define limite inferior y superior en rango de verde.
    lower_limmit = np.clip(bg_color_hsv - DELTA_BG, 0, 255)
    upper_limmit = np.clip(bg_color_hsv + DELTA_BG, 0, 255)

    return lower_limmit, upper_limmit


def area_of_interest(frame:np.array, lower_limit: np.array, upper_limit: np.array, coords: tuple = None)-> tuple[np.array, np.array, tuple]:
    '''
    Toma un frame y obtiene una máscara de color verde y su complementaria
    dentro de coordenadas de interés (si las recibe) a partir de un límite 
    inferior y superior dados.  
    ----------------------------------------------------------------
    Parámetros:
    ----------------------------------------------------------------
        -frame: frame del cual se obtiene la muestra.
        -upper_limmit: límite superior del color verde.
        -lower_limit: límite inferior del color verde.
        -coords: coordenadas de interés.
    -----------------------------------------------------------------
    Retorna:
    -----------------------------------------------------------------
        Si no se proporciona coordenadas:
        - Área de interés: frame recortado por las coordenadas del rectángulo
        verde obtenido de la máscara.
        - mask_not_bk_interés: máscara del área de interés con lo que no es 
        verde.
        -mas_bk_interés: máscara del área de interés con lo que es verde.

    -----------------------------------------------------------------
    Procedimiento:
    -----------------------------------------------------------------
        1. Sobre el frame en HSV aplica un filtro de forma tal que quede
            una máscara que sólo contenga valores True en aquellos píxeles
            que se encuentran entre lower_limit y upper_limit pasados. 
            A su vez, obtiene la complementaria de la máscara anterior,
            que vendría a ser todo aquello que no es fondo.
        2. Ordena las componentes conectadas de mayor a menor,
            sin contar el fondo, y se queda con la componente más grande,
            que vendría a representar el tablero de juego verde. (en caso de 
            que reciba coordenadas en el parámetro, se saltea este paso.)
        3. Con las coordenadas de dicha componente o las coordenadas pasadas
             recorta tanto el frame en RGB como la máscara de fondo y no fondo.
    '''
    


    
    
    frame_original_copy = frame.copy()
    frame_bg = cv2.cvtColor(frame_original_copy, cv2.COLOR_BGR2RGB)
    frame_hsv = cv2.cvtColor(frame_original_copy, cv2.COLOR_BGR2HSV)
    mask_bk = cv2.inRange(frame_hsv, lower_limit, upper_limit)
    
    mask_not_bk = cv2.bitwise_not(mask_bk)
    
    if coords:
        x, y, w, h = coords
    else:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bk, connectivity=8)
        areas = stats[:, cv2.CC_STAT_AREA]
        sorted_indices = np.argsort(areas)[::-1]
        sorted_stats = stats[sorted_indices]
        sorted_centroids = centroids[sorted_indices]
        sorted_stats = sorted_stats[1:]
        sorted_centroids = sorted_centroids[1:]
        fondo = sorted_indices[0]
        x, y, w, h = stats[fondo, cv2.CC_STAT_LEFT], stats[fondo, cv2.CC_STAT_TOP], stats[fondo, cv2.CC_STAT_WIDTH], stats[fondo, cv2.CC_STAT_HEIGHT]
    
    area_interes = frame_bg[y:y+h, x:x+w]
    mask_not_bk_interes = mask_not_bk[y:y+h, x:x+w]
    mask_bk_interes = mask_bk[y:y+h, x:x+w]

   
    if coords:
        return area_interes, mask_not_bk_interes, mask_bk_interes,
    else:
        return area_interes, mask_not_bk_interes, mask_bk_interes, (x, y, w, h)


def stop(frame_ant:np.array, frame_sig: np.array, threshold: int = 0)-> bool:
    '''
    Devuelve el True o False según si no se sobrepasa de un threshold de movimiento
    de un frame a otro.
    ----------------------------------------------------------------
    Parámetros:
    ----------------------------------------------------------------
        - frame_ant: frame_anterior.
        - frame_sig: frame siguiente.
        - threshold: umbral de movimiento.
    
    ----------------------------------------------------------------
    Retorna:
    ----------------------------------------------------------------
        -True: si no pasa el umbral de movimiento entre frame_ant
            y frame_sig.
        - False: si pasa del umbral de movimiento entre frame_ant
            y frame_sig.
    
    ----------------------------------------------------------------
    Procedimiento:
    ----------------------------------------------------------------
        1. Convierte a escala de grises los frames.
        2. Obtiene la diferencia entre frame_ant y frame_sig.
        3. Binariza la diferencia.
        4. Cuenta los píxeles en la diferencia.
        5. Devuelve True o False según la cantidad de píxeles.
    '''
    frame_ant_gray = cv2.cvtColor(frame_ant, cv2.COLOR_BGR2GRAY)
    frame_sig_gray = cv2.cvtColor(frame_sig, cv2.COLOR_BGR2GRAY)
    
   
    diff = cv2.absdiff(frame_ant_gray, frame_sig_gray)
    
   
    _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    num_diff_pixels = cv2.countNonZero(diff_thresh)

    return num_diff_pixels < threshold


def find_frame_stop(video_path: str = "")-> int:
    '''
    Esta función toma un video y devuelve el número de frame donde se 
    detiene el movimiento dentro de un área de interés predefinida.
    
    -----------------------------------------------------------------
    Parámetros:
    -----------------------------------------------------------------
        -video_path: ubicación del video.
    
    -----------------------------------------------------------------
    Retorna:
    -----------------------------------------------------------------
        -frame_stop: número de frame donde se detuvo el video en el
        área de interés.
    
    -----------------------------------------------------------------
    Procedimiento:
    -----------------------------------------------------------------
        1. Lee el video.
        2. Itera sobre los frames y compara el frame anterior con el 
            siguiente llamando a la función stop().
        3. Si el movimiento (cantidad de píxeles de diferencia) es menor
            a la proporción umbral (0.001 %), ese frame está detenido.
        4. Si el frame detenido presenta dos frames más sin mover luego,
            se considera que es el momento donde para el video y retorna
            el número de dicho frame.
    '''
   
    cap = cv2.VideoCapture(video_path)
    num_frame = 1
    frames_skip = 40
    movement = False
    consec_not_mov = 0
    frame_stop = 0
    ret, frame_ant = cap.read()
    

    while True:
        
        frame_ant_interest, mask_bk_frame_ant, mask_not_bk_frame_ant = area_of_interest(frame_ant, low_li, up_li, coords=coords_f)
    
        ret, frame_sig = cap.read()
        if not ret:
            break

        
        frame_sig_interest, mask_bk_frame_sig, mask_not_bk_frame_sig = area_of_interest(frame_sig, low_li, up_li, coords=coords_f)

        if num_frame > frames_skip:
            prop_mov = int(0.0001*frame_ant_interest.shape[0]*frame_ant_interest.shape[1])
            movement = stop(frame_ant_interest, frame_sig_interest, threshold=prop_mov)
            
            if movement:
                
                frame_stop = num_frame
                consec_not_mov += 1
            else:
                consec_not_mov = 0
                
            
            # Si pasan 2 frames consecutivos sin movimiento, considerar que los dados se han detenido
            if consec_not_mov >= 2:
                
                break 
        frame_ant = frame_sig
        
        num_frame += 1
        

    
    cap.release()
    cv2.destroyAllWindows()
    
    
    return frame_stop

def find_dados(frame_C:np.array,mask_not_bk: np.array)-> list[tuple]:
    '''
    Obtiene los rectángulos de las componentes conectadas presentes
    en la máscara no fondo de un frame dado y devuelve aquellos que por su forma considera que son 
    dados.
    -----------------------------------------------------------------
    Parámetros:
    -----------------------------------------------------------------
        - frame_c: frame en RGB.
        - mask_not_bk: máscara que no es fondo del frame.
    
    -----------------------------------------------------------------
    Retorna:
    -----------------------------------------------------------------
        - cuadrados: lista con las coordenadas (x,y,w,h) de los cuadrados.

    -----------------------------------------------------------------
    Procedimiento:
    -----------------------------------------------------------------
        1. Aplica apertura sobre la máscara mask_not_bk para dividir
            posibles dados juntos.
        2. Aplica blur sobre la máscara modificada.
        3. Binariza nuevamente la máscara para eliminar definitivamente
            uniones no deseadas.
        4. Obtiene las componentes conectadas de la nueva máscara.
        5. Si la componente no cumple con ciertas medidas de ancho y alto
            no es consideradas un dado, caso contrario, la agrega a la lista
            de dados.

    '''
    cuadrados = []
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    mask_open = cv2.morphologyEx(mask_not_bk, cv2.MORPH_OPEN, k, iterations= 2)
    mask_blur = cv2.GaussianBlur(mask_open, (5, 5), 0)
    _, mask_blur_binary = cv2.threshold(mask_blur, 170, 255, cv2.THRESH_BINARY)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_blur_binary, connectivity=4)
    for i in range(1, num_labels):
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if h > (0.2* frame_C.shape[0]) or w > (0.2* frame_C.shape[1]) or w < (0.05 * frame_C.shape[1]):
            continue
        cuadrado = frame_C[y:y+h, x:x+w]
        cuadrados.append((x,y,w,h))
        
    return cuadrados


def contar_puntos(frame: np.array, lista_dados: list)-> tuple[dict, int]:
    '''
    Cuenta los puntos de cada dado dentro de un frame.
    -----------------------------------------------------------------
    Parámetros:
    -----------------------------------------------------------------
        - frame: frame en RGB.
        - lista_dados: lista de dados en un frame.
    
    -----------------------------------------------------------------
    Retorna:
    -----------------------------------------------------------------
        - dict_cuadrados_puntos: diccionario con clave número de dado
          y valor puntaje del dado.
        - puntos_totales: puntos totales entre todos los dados.
    
    -----------------------------------------------------------------
    Procedimiento:
    -----------------------------------------------------------------
        Para cada dado en la lista:
        1.Se obtienen sus valores blancos
          aplicando un filtro de rangos de dicho color.
        2. SObre esa máscara de blanco, se palica un blur para descartar
           valores blancos que no son un putno del dado.
        3. Se binariza la máscara modificada.
        4. Se obtienen las componentes conectadas y si cumple con ciertas
            medidas, es considerada un punto y se la agrega al puntaje del 
            dado.
        5. Se agrega el número del dado y el puntaje total al diccionario.

    
    '''
    puntos_totales = 0
    dict_cuadrados_puntos = {}
    for i in range(len(lista_dados)):
        x,y,w,h = lista_dados[i]
        cuadrado = frame[y:y+h, x:x+w]
        cuadrado_hsv = cv2.cvtColor(cuadrado, cv2.COLOR_RGB2HSV)
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 50, 255])
        mask_white = cv2.inRange(cuadrado_hsv, lower_white, upper_white)
        mask_white_blur = cv2.GaussianBlur(mask_white, (11,11), 0)
        _, mask_white_blur_bin = cv2.threshold(mask_white_blur, 160, 255, cv2.THRESH_BINARY)

        num_labels_c, labels_c, stats_c, centroids_c = cv2.connectedComponentsWithStats(mask_white_blur_bin, connectivity=8)
        
        cuadrado_draw = cuadrado.copy()
        puntos = 0
        
        
        for k in range(1, num_labels_c):  
            area_c = stats_c[k, cv2.CC_STAT_AREA]
            x_p, y_p, w_p, h_p = stats_c[k, cv2.CC_STAT_LEFT], stats_c[k, cv2.CC_STAT_TOP], stats_c[k, cv2.CC_STAT_WIDTH], stats_c[k, cv2.CC_STAT_HEIGHT]
            if h_p < (0.3 * cuadrado_draw.shape[0]) and w_p < (0.3 * cuadrado_draw.shape[1]):
                cx, cy = int(centroids_c[k][0]), int(centroids_c[k][1])
                cv2.circle(cuadrado_draw, (cx, cy), 2, (0, 255, 0), -1)
                puntos += 1
            dict_cuadrados_puntos[(x,y,w,h)] = puntos
        puntos_totales += puntos
        
        
    return dict_cuadrados_puntos, puntos_totales
                
def generala(combinaciones: list[int])-> tuple[int,str]:
    '''
    Detecta la jugada obtenida (o no) de una tirada mediante
    el análisis de los puntos de los dados en la misma.
    -----------------------------------------------------------------
    Parámetros:
    -----------------------------------------------------------------
        - combinaciones: lista de enteros que representan los valores
        de los dados.
    
    -----------------------------------------------------------------
    Retorna:
    -----------------------------------------------------------------
        - puntuacion_generala: puntuación final de la jugada.
        - text_combinación: texto relacionada a la jugada o no obtenida.
    
    -----------------------------------------------------------------
    Procedimiento:
    -----------------------------------------------------------------
        1. Analiza las combianciones y si entran dentro de la condición
           de una jugada, se la adosa a la misma.

    
    
    '''
    combinaciones.sort()  
    puntuacion_generala = 0
    text_combinacion = "Sin combinación"
    if len(combinaciones) == 5:
        # Generala
        if len(set(combinaciones)) == 1:
            puntuacion_generala = 50
            text_combinacion = "Generala: 50 puntos(5 dados iguales)"
        # Poker
        elif any(combinaciones.count(val) == 4 for val in combinaciones):
            puntuacion_generala = 40
            text_combinacion = "Poker: 40 puntos(4 dados iguales)"
        # Full
        elif any(combinaciones.count(val) == 3 for val in combinaciones):
            puntuacion_generala = 30
            text_combinacion = "Full: 30 puntos(3 dados iguales)"
        # Escalera
        elif combinaciones == [1, 2, 3, 4, 5]:
            puntuacion_generala = 20
            text_combinacion = "Escalera menor: 20 puntos"
        elif combinaciones == [2, 3, 4, 5, 6]:
            puntuacion_generala = 25
            text_combinacion = "Escalera mayor: 25 puntos"
        # Suma de números específicos
        else:
            puntuacion_generala = sum(combinaciones)  # Sumar los valores de los dados
            text_combinacion = f"puntaje: {puntuacion_generala} puntos. No forma juego"
    return puntuacion_generala, text_combinacion
            

def dibujar_dados(frame: np.array, lista_dados: list = None, dic_d: dict = None, label: bool = False, puntaje_jugada: int = 0, texto: str = "Suma: 0 puntos")-> np.array:
    '''
    Dibuja los dados de acuerdo a si se pasa una lista_dados o un dic_d.
    -----------------------------------------------------------------
    Parámetros:
    -----------------------------------------------------------------
        - frame: frame en RGB.
        - lista_dados: lista de dados.
        - dic_d: diccionario dado-valor.
        - label: indica si se quiere o no dibujar los labels en caso de
            que se pase un dic_d.
        - puntaje_jugada: puntaje de la jugada si es que se pasa.
        - texto: texto con el puntaje de la jugada.
    
    -----------------------------------------------------------------
    Retorna:
    -----------------------------------------------------------------
        - frame_dibujado: frame dibujado correspondiente con lo solicitado.
    
    -----------------------------------------------------------------
    Procedimiento:
    -----------------------------------------------------------------
        **Si hay diccionario:**
            Para cada dado:
                1. Agrega a la lista de combinaciones el puntaje y escribe el 
                    valor del dado para cada dado en caso de que label == True.
            2. Obtiene la puntuación de la generala llamando a la función 
                genreala() y pasándole como argumento combinaciones.
            3. Escribe el resultado de la jugada.
        
        ** Si no hay diccionario.**
            Para cada dado en lista_dados:
                1. Dibuja su rectángulo contenedor.
            2. Si se paso el puntaje de la jugada, se lo escribe.
        
    '''
    
    frame_draw = frame.copy()
    combinaciones = []
    puntuacion_generala = 0

    if dic_d is not None:
        i = 0
        for dado, punto in dic_d.items():
            x, y, w, h = dado
            combinaciones.append(punto)  # Agregar el valor del dado al análisis de combinaciones
            cv2.rectangle(frame_draw, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if label: 
                text_dado = f'Dado {i+1}'
                text_puntaje = f'Puntaje:{punto}'
                cv2.putText(frame_draw, text_dado, (x - 50, y - 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
                cv2.putText(frame_draw, text_puntaje, (x - 50, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
                i += 1
        
        puntuacion_generala, text_combinacion = generala(combinaciones)

        if label:
            text_puntaje = f'Puntaje total: {puntuacion_generala}'
            cv2.putText(frame_draw, text_combinacion, (20, 60), cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 255, 0), thickness=2)

    else:
        for i in range(len(lista_dados)):
            x, y, w, h = lista_dados[i]
            cv2.rectangle(frame_draw, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if puntaje_jugada != 0:
            text_puntaje = f'Puntaje total: {puntaje_jugada}'
            cv2.putText(frame_draw, texto, (20, 60), cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 255, 0), thickness=2)

    return frame_draw

def unir_frames(frame_original:np.array, area_dibujada:np.array, coords: tuple)-> np.array:
    '''
    Une el frame original con el frame dibujado.
    -----------------------------------------------------------------
    Parámetros:
    -----------------------------------------------------------------
        - frame_original: frame RGB sobre el cual se quiere unir el 
            área dibujada.
        - area_dibujada: área dibujada.
        - coords: coordenadas para unir sobre el frame original el área
            dibujada.
    
    -----------------------------------------------------------------
    Retorna:
    -----------------------------------------------------------------
        -frame_dibujado: frame original con el área dibujada.
    '''
    
    frame_dibujado = frame_original.copy()
    x, y, w, h = coords
    frame_dibujado[y:y+h, x:x+w] = area_dibujada
    return frame_dibujado



def modificar_frames(video_path: str, frame_stop: int, low_limit: np.array, up_limit: np.array, coords: tuple, new_w: int, new_h:int) -> list[np.array]:
    '''
    Modifica los frames según el momento de la jugada en que se encuentren.
    Muestra el video modificado.
    -----------------------------------------------------------------
    Parámetros:
    -----------------------------------------------------------------
        - video_path: path del video que contiene la jugada.
        - frame_stop: número de frame donde se detienen los dados.
        - up_limit: límite superior de verde.
        - coords: coordenadas del área de interés del video.
        - new_h: nueva altura del video (sólo para fines de visualización).
        - new_w: nuevo ancho del video (sólo para fines de visualización).
    
    -----------------------------------------------------------------
    Retorna:
    -----------------------------------------------------------------
        - frames_mod : lista que contiene los frames dibujados.
        - dic_dados: diccionario dado-valor.
        - puntaje_generala: puntaje obtenido de la jugada.
        - total_puntos: puntaje total de los dados.
    
    -----------------------------------------------------------------
    Procedimiento:
    -----------------------------------------------------------------
        1. Lee el video e itera sobre los frames.
            a. Si num_frame es menor que frame_stop, dibuja sólo los
            rectángulos contenedores de los dados.
            b. Si num_frame es igual a frame_stop, obtiene la lista_dados,
                luego el dic_dados y puntaje_total y con ellos dibuja
                los rectángulos de los dados, los puntos de los mismos y 
                escribe el resultado de la jugada de la generala.
            c. Pasado num_frame a frame_stop: continúa dibujando
                los valores obtenidos en frame_stop hasta que la cantidad
                de dados sea menor a 5, caso en el cual sólo dibuja los
                rectángulos contenedores de los dados y el texto resultado
                de la jugada.
        2. Muestra el video modificado.



    
    '''
    
    
    cap = cv2.VideoCapture(video_path)
    ret, frame_ant = cap.read()
    if not ret:
        print("No se pudo leer el video.")
        return []

    # Inicialización de propiedades del video
    num_frame = 0
    frames_mod = []
    total_puntos = 0
    move = False

    while True:
        ret, frame_sig = cap.read()
        if not ret:
            break

        # Procesamiento según la lógica del frame actual
        if num_frame < frame_stop:
            area_interes_f_n, mask_not_bk_f_n, mask_bk_f_n = area_of_interest(
                frame_sig, lower_limit=low_limit, upper_limit=up_limit, coords=coords
            )
            area_interes_f_n = cv2.cvtColor(area_interes_f_n, cv2.COLOR_BGR2RGB)
            lista_dados = find_dados(area_interes_f_n, mask_not_bk_f_n)
            area_dib = dibujar_dados(area_interes_f_n, lista_dados=lista_dados)
            frame_dib = unir_frames(frame_sig, area_dib, coords=coords)

        elif num_frame == frame_stop:
            area_interest_frame_stop, mask_not_bk_frame_stop, mask_bk_frame_stop = area_of_interest(
                frame_sig, lower_limit=low_limit, upper_limit=up_limit, coords=coords
            )
            lista_dados = find_dados(area_interest_frame_stop, mask_not_bk_frame_stop)
            dic_dados, puntaje = contar_puntos(area_interest_frame_stop, lista_dados)
            combinaciones = [punto for _, punto in dic_dados.items()]
            puntaje_generala, texto_ = generala(combinaciones)
            texto = texto_
            total_puntos += puntaje

            area_interest_frame_stop = cv2.cvtColor(area_interest_frame_stop, cv2.COLOR_BGR2RGB)
            area_dib = dibujar_dados(area_interest_frame_stop, dic_d=dic_dados, label=True, puntaje_jugada=puntaje)
            frame_dib = unir_frames(frame_sig, area_dib, coords=coords)

        elif frame_stop < num_frame and not move:
            area_interest_frame_stop, mask_not_bk_frame_stop, mask_bk_frame_stop = area_of_interest(
                frame_sig, lower_limit=low_limit, upper_limit=up_limit, coords=coords
            )
            lista_dados = find_dados(area_interest_frame_stop, mask_not_bk_frame_stop)
            if len(lista_dados) < 5:
                move = True
                continue
            area_interest_frame_stop = cv2.cvtColor(area_interest_frame_stop, cv2.COLOR_BGR2RGB)
            area_dib = dibujar_dados(area_interest_frame_stop, dic_d=dic_dados, label=True, puntaje_jugada=puntaje)
            frame_dib = unir_frames(frame_sig, area_dib, coords=coords)

        if move:
            area_interes_f_n, mask_not_bk_f_n, mask_bk_f_n = area_of_interest(
                frame_sig, lower_limit=low_limit, upper_limit=up_limit, coords=coords
            )
            lista_dados = find_dados(area_interes_f_n, mask_not_bk_f_n)
            area_interes_f_n = cv2.cvtColor(area_interes_f_n, cv2.COLOR_BGR2RGB)
            area_dib = dibujar_dados(area_interes_f_n, lista_dados=lista_dados, puntaje_jugada=puntaje_generala, texto=texto)
            frame_dib = unir_frames(frame_sig, area_dib, coords=coords)

        num_frame += 1
        frame_dib_resize = cv2.resize(frame_dib, (new_w, new_h))
        cv2.imshow(winname='', mat=frame_dib_resize)

        # Salir si el usuario presiona 'q'
        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("Detenido por el usuario.")
            break

        frames_mod.append(frame_dib)

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

    print("Procesamiento completado.")
    return frames_mod, dic_dados, puntaje_generala, total_puntos

def guardar_video(lista_frames: list, output_path: str, fps=30, codec="mp4v")->None:
    '''
    Guarda un video en 30 fps y formato mp4 a partir de una lista de frames.
    -----------------------------------------------------------------
    Parámetros:
    -----------------------------------------------------------------
        - lista_frames: lista de frames a partir de la cual se arma el video.
        - output_path: ubicación donde guardar el video.
        - fps: fps del video, seteado a 30.
        - codec: codificación del video, seteada a "mp4v" (.mp4)
    
    -----------------------------------------------------------------
    Retorna:
    -----------------------------------------------------------------
        -None.
    
    -----------------------------------------------------------------
    Procedimiento:
    -----------------------------------------------------------------
        1. Si el output_path no está creado, lo crea.
        2. Crea el video
        3. Guarda el video en el output_path.
    
    '''
    
    
    if not lista_frames:
        raise ValueError("La lista de frames está vacía. No se puede crear un video.")

    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    
    height, width, _ = lista_frames[0].shape

    
    fourcc = cv2.VideoWriter_fourcc(*codec) 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Escribir cada frame en el archivo de video
    for frame in lista_frames:
        out.write(frame)

    
    out.release()

    print(f"Video guardado correctamente en: {output_path}")


def exec(path: str ="", output_video_path: str =None)->None:
    '''
    Ejecuta las funciones necesarias para detectar la jugada y dibujar
    el video.
    -----------------------------------------------------------------
    Parámetros:
    -----------------------------------------------------------------
        - path: path del video.
        - output_video_path: ubicación de salida del video.
    
    -----------------------------------------------------------------
    Retorna:
    -----------------------------------------------------------------
        -None.
    
    -----------------------------------------------------------------
    Procedimiento:
    -----------------------------------------------------------------
        1. Obtiene el frame_stop.
        2. Obtiene los frames_modificados, dic_dados, puntaje y total_puntos.
        3. Si se indicó un output_video, se guarda el video.
        4. Imprime por consola el resultado de la jugada.
    '''

        #imshow(mask_not_bk_f, blocking= True)
    frame_stop = find_frame_stop(path)

    new_width = int(area_interest_f.shape[1] * 0.4)
    new_height = int(area_interest_f.shape[0]  * 0.4)
    video_path = path

    frames_modificados, dic_dados, puntaje, total_puntos = modificar_frames(video_path, frame_stop= frame_stop, low_limit= low_li, up_limit = up_li,coords= coords_f, new_w = new_width, new_h = new_height)

    if output_video_path != None:
        guardar_video(frames_modificados, output_path=output_video_path)

    print(f"Esta tirada tuvo los siguientes puntos:")
    s = 1
    for dado, punto in dic_dados.items():
        print(f"El dado {s}: {punto}")
        s+=1
    print(f"El total de puntos es: {total_puntos}")
    print(f"El puntaje total fue: {puntaje}")

if __name__ == "__main__":
    path = input("Ingrese la ubicación del archivo: ")
    output = input("Ingrese la ubicación donde desea guardar el video. Si no desea guardarlo, presione enter: ")
    print(f"Procesando video {path} ...")

    cap = cv2.VideoCapture(path)
    ret, frame_test = cap.read()

    cap.release()
    cv2.destroyAllWindows()

    global low_li
    global up_li
    global coords_f
    low_li, up_li = green_sample(frame_test)
    area_interest_f, mask_not_bk_f, mask_bk_f, coords_f = area_of_interest(frame_test, lower_limit= low_li, upper_limit= up_li)
        
    exec(path, output)


