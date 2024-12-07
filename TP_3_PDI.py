from rqst import *

def green_sample(frame)-> tuple[np.array, np.array,np.array]:
    '''
    Toma un frame y realiza una muesta de color verde a partir de la
    cual devuelve límites inferiores y superiores de rango de dicho color
    que sirve para hacer máscaras.  
    ----------------------------------------------------------------
    Parámetros:
    ----------------------------------------------------------------
        -Frame: frame del cual se obtiene la muestra.
    
    -----------------------------------------------------------------
    Retorna:
    -----------------------------------------------------------------
        -upper_limmit: límite superior del color verde.
        -lower_limit: límite inferior del color verde.
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
        -Frame: frame del cual se obtiene la muestra.
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
    
    frame_ant_gray = cv2.cvtColor(frame_ant, cv2.COLOR_BGR2GRAY)
    frame_sig_gray = cv2.cvtColor(frame_sig, cv2.COLOR_BGR2GRAY)
    
   
    diff = cv2.absdiff(frame_ant_gray, frame_sig_gray)
    
   
    _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    num_diff_pixels = cv2.countNonZero(diff_thresh)

    return num_diff_pixels < threshold


def find_frame_stop(video_path: str = "")-> int:
   
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

def find_dados(frame_C,mask_not_bk: np.array)-> list:
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


def contar_puntos(frame, lista_dados: np.array)-> tuple[dict, int]:
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
                
def geberala(combinaciones=list[int])-> tuple[int,str]:
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
        
        puntuacion_generala, text_combinacion = geberala(combinaciones)

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
    frame_dibujado = frame_original.copy()
    x, y, w, h = coords
    frame_dibujado[y:y+h, x:x+w] = area_dibujada
    return frame_dibujado



def modificar_frames(video_path, frame_stop, low_limit, up_limit, coords, new_w, new_h) -> list:
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
            puntaje_generala, texto_ = geberala(combinaciones)
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

def guardar_video(lista_frames, output_path, fps=30, codec="mp4v"):
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


def user(path="", output_video_path =None):
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
    print(f"Procesando video {path} ...")
    output = f'output\{path}'
    cap = cv2.VideoCapture(path)
    ret, frame_test = cap.read()

    cap.release()
    cv2.destroyAllWindows()

    global low_li
    global up_li
    global coords_f
    low_li, up_li = green_sample(frame_test)
    area_interest_f, mask_not_bk_f, mask_bk_f, coords_f = area_of_interest(frame_test, lower_limit= low_li, upper_limit= up_li)
        
    user(path, output)


