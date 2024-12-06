from rqst import *
cap = cv2.VideoCapture("data/tirada_1.mp4")
ret, frame_prueba = cap.read()
cap.release()
cv2.destroyAllWindows()

frame_rgb = cv2.cvtColor(frame_prueba, cv2.COLOR_BGR2RGB)
bg_crop_RGB = frame_rgb[100:200, 100:200, :]
#mshow(bg_crop_RGB)                     # Cortar una porcion correspondiente al fondo.
#imshow(bg_crop_RGB, title="Frame 1 - Crop fondo verde")
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

def dibujar_cuadrados(frame, mask, ret_cuadrados: bool = None, cuadrados_draw:list = False, ret_puntos: bool = None, label_cuadrados : bool = False, puntos_totales_dados: int = False, lista_puntos: list = False, puntos_totales_draw: bool = False) -> np.array:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    if cuadrados_draw:
        num_labels = len(cuadrados_draw)
    if not lista_puntos:
        lista_puntos = []
    frame_draw = frame.copy()
    cuadrados = []
    c = 0
    for i in range(1, num_labels):
        if cuadrados_draw:
            x,y,w,h = cuadrados_draw[i]
        else:    
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        if h > (0.1 * frame_draw.shape[0]) or w > (0.1 * frame_draw.shape[1]) or w < (0.05 * frame_draw.shape[1]):
            continue
        
        cuadrados.append((x,y,w,h))
        c += 1
        
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        cv2.circle(frame_draw, (cx, cy), 2, (0, 255, 0), -1)
        cv2.rectangle(frame_draw, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if ret_puntos:
            cuadrado = frame[y:y+h, x:x+w]
            cuadrado_hsv = cv2.cvtColor(cuadrado, cv2.COLOR_RGB2HSV)

            lower_white = np.array([0, 0, 150])
            upper_white = np.array([180, 50, 255])
            mask_white = cv2.inRange(cuadrado_hsv, lower_white, upper_white)

            mask_white_blur = cv2.GaussianBlur(mask_white, (7,7), 0)
            _, mask_white_blur_bin = cv2.threshold(mask_white_blur, 150, 255, cv2.THRESH_BINARY)

            #imshow(mask_white_blur_bin, blocking=True, title='Valores blancos encontrados')

            num_labels_c, labels_c, stats_c, centroids_c = cv2.connectedComponentsWithStats(mask_white_blur_bin, connectivity=8)
            
            cuadrado_draw = cuadrado.copy()
            puntos = 0
            puntos_totales = 0
            
            for k in range(1, num_labels_c):  # Cambié el índice a 'k' para evitar conflicto
                area_c = stats_c[k, cv2.CC_STAT_AREA]
                x_p, y_p, w_p, h_p = stats_c[k, cv2.CC_STAT_LEFT], stats_c[k, cv2.CC_STAT_TOP], stats_c[k, cv2.CC_STAT_WIDTH], stats_c[k, cv2.CC_STAT_HEIGHT]
                if h_p < (0.3 * cuadrado_draw.shape[0]) and w_p < (0.3 * cuadrado_draw.shape[1]):
                    # Dibujar el centroide
                    cx, cy = int(centroids_c[k][0]), int(centroids_c[k][1])
                    cv2.circle(cuadrado_draw, (cx, cy), 2, (0, 255, 0), -1)
                    puntos += 1
            #imshow(cuadrado_draw)
           
            if label_cuadrados:
                label_text = f'Cuadrado {c}'
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                x_offset = max(x - 15, 0)
                y_offset = max(y - 20, text_size[1])  # Asegurarse de que el texto no se dibuje fuera
                cv2.putText(frame_draw, label_text, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 1, cv2.LINE_AA)
                
                # Verificar si la lista de puntos tiene suficientes elementos antes de acceder
                if len(lista_puntos) > c - 1:
                    label_text = f'puntos {lista_puntos[c-1]}'
                else:
                    label_text = f'puntos {puntos}'
                    lista_puntos.append(puntos)
                y_offset = max(y - 15, text_size[1] * 2)  # Asegurar que no se sobrepongan
                cv2.putText(frame_draw, label_text, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255))
            puntos_totales += puntos
    
    if puntos_totales_draw:
        if puntos_totales_dados:
            puntos_totales = puntos_totales_dados
        else:
            label_text = f'Puntos totales: {puntos_totales}'
        cv2.putText(frame_draw, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)

    if ret_cuadrados and ret_puntos:
        return frame_draw, cuadrados, puntos_totales, lista_puntos
    else:
        return frame_draw

    

    



def area_of_interest(frame_original: np.array, frame_bg: np.array, stop: bool = False, initial: bool = False, coords: tuple = (), puntos_totales_dados = None, first: bool = False, cuadrados_draw: list = None, lista_puntos: list = None) -> np.array:
    frame_original_copy = frame_original.copy()
    frame_hsv = cv2.cvtColor(frame_bg, cv2.COLOR_RGB2HSV)
    mask_bk = cv2.inRange(frame_hsv, lower_limmit, upper_limmit)
    mask_not_bk = cv2.bitwise_not(mask_bk)

    bg = cv2.bitwise_and(frame_bg, frame_bg, mask=mask_bk)
    not_bk = cv2.bitwise_and(frame_bg, frame_bg, mask=mask_not_bk)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bk, connectivity=8)

    areas = stats[:, cv2.CC_STAT_AREA]
    sorted_indices = np.argsort(areas)[::-1]
    sorted_stats = stats[sorted_indices]
    sorted_centroids = centroids[sorted_indices]

    sorted_stats = sorted_stats[1:]
    sorted_centroids = sorted_centroids[1:]

    fondo = sorted_indices[0]

    if initial:
        x_area_interes, y_area_interes, w_area_interes, h_area_interes = stats[fondo, cv2.CC_STAT_LEFT], stats[fondo, cv2.CC_STAT_TOP], stats[fondo, cv2.CC_STAT_WIDTH], stats[fondo, cv2.CC_STAT_HEIGHT]
        area_interes = frame_bg[y_area_interes:y_area_interes+h_area_interes, x_area_interes:x_area_interes+w_area_interes]
        mask_not_bk_interes = mask_not_bk[y_area_interes:area_interes.shape[0], x_area_interes:area_interes.shape[1]]
    else:
        area_interes = frame_bg
        x_area_interes, y_area_interes, w_area_interes, h_area_interes = coords
        mask_not_bk_interes = mask_not_bk[y_area_interes:y_area_interes + h_area_interes, x_area_interes:x_area_interes + w_area_interes]

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    mask_open = cv2.morphologyEx(mask_not_bk_interes, cv2.MORPH_OPEN, k)
    mask_blur = cv2.GaussianBlur(mask_open, (5, 5), 0)
    _, mask_blur_binary = cv2.threshold(mask_blur, 170, 255, cv2.THRESH_BINARY)

    img_draw = area_interes.copy()

    if stop and first:
        img_draw, cuadrados_f, puntos_totales_f, lista_puntos_f = dibujar_cuadrados(area_interes, mask_blur_binary, ret_cuadrados=True, ret_puntos=True, label_cuadrados=True, puntos_totales_draw=True)
    elif stop and not first:
        img_draw = dibujar_cuadrados(area_interes, mask_blur_binary, cuadrados_draw=cuadrados_draw, label_cuadrados=True, puntos_totales_dados=puntos_totales_dados, lista_puntos=lista_puntos, puntos_totales_draw=True, ret_puntos=True)
    else:
        img_draw = dibujar_cuadrados(area_interes, mask_blur_binary)

    if img_draw.shape[:2] != (h_area_interes, w_area_interes):
        img_draw = cv2.resize(img_draw, (w_area_interes, h_area_interes), interpolation=cv2.INTER_LINEAR)

    frame_original_copy[y_area_interes:y_area_interes + h_area_interes, x_area_interes:x_area_interes + w_area_interes] = img_draw

    if initial:
        return area_interes, (x_area_interes, y_area_interes, w_area_interes, h_area_interes), frame_original_copy
    if stop:
        if first:
            return frame_original_copy, cuadrados_f, puntos_totales_f, lista_puntos_f
        return frame_original_copy
    else:
        return area_interes, frame_original_copy



def detectar_movimiento(video_path, umbral_movimiento=5000, escala=0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("No se pudo abrir el video.")
        return

    ret, frame_actual = cap.read()
    if not ret:
        print("No se pudo leer el primer frame.")
        cap.release()
        return

    frames_insertar = {}
    frame_actual, coords, frame_dibujado = area_of_interest(cv2.cvtColor(frame_actual, cv2.COLOR_BGR2RGB), cv2.cvtColor(frame_actual, cv2.COLOR_BGR2RGB), initial=True)
    x, y, w, h = coords
    frame_number = 0
    frames_insertar[frame_number] = frame_dibujado
    paro = False
    cant_no_parados = 0
    height, width = frame_actual.shape[:2]
    scale_x = 800 / float(width)
    scale_y = 600 / float(height)
    scale = min(scale_x, scale_y)
    new_width = int(width * scale)
    new_height = int(height * scale)
    cuadrados_f = []
    puntos_totales_f = 0
    lista_puntos_f = []
    primer_frame_detectado = False

    while cap.isOpened():
        ret, frame_siguiente_original = cap.read()
        if not ret or type(frame_siguiente_original) == None:
            print("Fin del video.")
            break

        frame_siguiente_original = cv2.cvtColor(frame_siguiente_original, cv2.COLOR_BGR2RGB)
        frame_siguiente = frame_siguiente_original[y:y + h, x:x + w]

        diferencia = cv2.absdiff(frame_actual, frame_siguiente)
        _, diferencia_bin = cv2.threshold(diferencia, 30, 255, cv2.THRESH_BINARY)

        num_pixeles_movimiento = np.sum(diferencia_bin > 0)

        a, frame_dibujado = area_of_interest(frame_siguiente_original, frame_siguiente, coords=coords)

        if num_pixeles_movimiento != 0 and paro:
            cant_no_parados += 1
            frame_dibujado = area_of_interest(
                frame_siguiente_original,
                frame_siguiente,
                stop=True,
                coords=coords,
                puntos_totales_dados=puntos_totales_f,
                cuadrados_draw=cuadrados_f,
                lista_puntos=lista_puntos_f
            )

            if cant_no_parados > 20:
                frames_insertar[frame_number] = frame_dibujado
                frame_redimensionado = cv2.resize(frame_dibujado, (new_width, new_height))
                cv2.imshow('Frame movido', frame_redimensionado)
                return

        elif num_pixeles_movimiento == 0 and frame_number > 50:
            cant_no_parados = 0
            paro = True

            if not primer_frame_detectado:
                frame_dibujado, cuadrados_f, puntos_totales_f, lista_puntos_f = area_of_interest(
                    frame_siguiente_original,
                    frame_siguiente,
                    stop=True,
                    coords=coords,
                    first=True
                )
                primer_frame_detectado = True
            else:
                frame_dibujado = area_of_interest(
                    frame_siguiente_original,
                    frame_siguiente,
                    stop=True,
                    coords=coords,
                    puntos_totales_dados=puntos_totales_f,
                    lista_puntos=lista_puntos_f,
                    cuadrados_draw=cuadrados_f
                )

        frames_insertar[frame_number] = frame_dibujado
        frame_redimensionado = cv2.resize(frame_dibujado, (new_width, new_height))
        cv2.imshow( 'Frame nada',  frame_redimensionado)

        frame_number += 1
        frame_actual = frame_siguiente

        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("Detenido por el usuario.")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return frames_insertar



for i in range(1,5):
    video_path = f"data/tirada_{i}.mp4"

    frames_insertar = detectar_movimiento(video_path,  escala=0.5)


#     _, frame_bin = cv2.threshold(frame, thresh=80, maxval=255, type=cv2.THRESH_BINARY)

#     imagen,dados = dado_imagen(frame)

#     imshow(imagen,blocking=True)

# frame_c, circulos = find_and_draw_circles(frame_bin,dp_ = 1.2, minD=7,minR=2,p1=1,p2=8,maxR=8)

# ################################## Análisis Fondo Verde ##############################
# imshow(frame_hsv)
# frame_bg = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2RGB)
# imshow(frame_bg)
# # Tomo una muestra
# bg_crop_RGB = frame_bg[30:200, 30:200, :]                     # Cortar una porcion correspondiente al fondo.
# imshow(bg_crop_RGB, title="Frame 1 - Crop fondo verde")
# img_pixels = bg_crop_RGB.reshape(-1,3)
# colours, counts = np.unique(img_pixels, axis = 0, return_counts= True)
# n_colours = colours.shape[0]
# n_colours
# colours[0]

# bg_color_rgb = colours[4]
# bg_color_rgb = np.reshape(bg_color_rgb, (1,1,-1))
# bg_color_hsv = cv2.cvtColor(bg_color_rgb, cv2.COLOR_RGB2HSV)

# # ----------------------------------------------------------------------------
# # --- Prueba 1: Match Exacto -------------------------------------------------
# # ----------------------------------------------------------------------------
# mask_bk = cv2.inRange(frame_hsv, bg_color_hsv[0,0,:], bg_color_hsv[0,0,:])
# mask_person = cv2.bitwise_not(mask_bk)
# bg = cv2.bitwise_and(frame_bg, frame_bg, mask=mask_bk)
# person = cv2.bitwise_and(frame_bg, frame_bg, mask=mask_person)

# plt.figure()
# ax = plt.subplot(231); imshow(frame_bg, title="Frame", new_fig=False)
# plt.subplot(232, sharex=ax, sharey=ax), imshow(mask_bk, title="Máscara Background", new_fig=False)
# plt.subplot(233, sharex=ax, sharey=ax), imshow(mask_person, title="Máscara persona", new_fig=False)
# plt.subplot(235, sharex=ax, sharey=ax), imshow(bg, title="Background", new_fig=False)
# plt.subplot(236, sharex=ax, sharey=ax), imshow(person, title="Persona", new_fig=False)
# plt.suptitle(f'bg_RGB = {bg_color_rgb} | bg_HSV = {bg_color_hsv}')
# plt.show(block=False)

# # Analizemos en HSV...
# bg_HSV = cv2.bitwise_and(frame_hsv, frame_hsv, mask=mask_bk)
# person_HSV = cv2.bitwise_and(frame_hsv, frame_hsv, mask=mask_person)
# plt.figure()
# ax = plt.subplot(231); imshow(frame_hsv, title="Frame", new_fig=False)
# plt.subplot(232, sharex=ax, sharey=ax), imshow(mask_bk, title="Máscara Background", new_fig=False)
# plt.subplot(233, sharex=ax, sharey=ax), imshow(mask_person, title="Máscara persona", new_fig=False)
# plt.subplot(235, sharex=ax, sharey=ax), imshow(bg_HSV, title="Background", new_fig=False)
# plt.subplot(236, sharex=ax, sharey=ax), imshow(person_HSV, title="Persona", new_fig=False)
# plt.subplot(234, sharex=ax, sharey=ax), imshow(frame_bg, title="Frame RGB", new_fig=False)
# plt.suptitle(f'HSV - bg_RGB = {bg_color_rgb} | bg_HSV = {bg_color_hsv}')
# plt.show(block=False)

# # ----------------------------------------------------------------------------
# # --- Prueba 2: Rango --------------------------------------------------------
# # ----------------------------------------------------------------------------
# delta_bg = np.array([10.0, 150.0, 150.0])
# # delta_bg = np.array([0, 0, 0])
# lower_limmit = np.clip(bg_color_hsv - delta_bg, 0, 255)
# upper_limmit = np.clip(bg_color_hsv + delta_bg, 0, 255)
# mask_bk = cv2.inRange(frame_hsv, lower_limmit, upper_limmit)
# mask_not_bk = cv2.bitwise_not(mask_bk)
# bg = cv2.bitwise_and(frame, frame, mask=mask_bk)
# not_bk = cv2.bitwise_and(frame, frame, mask=mask_not_bk)

# plt.figure()
# ax = plt.subplot(231); imshow(frame, title="Frame", new_fig=False)
# plt.subplot(232, sharex=ax, sharey=ax), imshow(mask_bk, title="Mascara bg", new_fig=False)
# plt.subplot(233, sharex=ax, sharey=ax), imshow(mask_not_bk, title="Mascara not_bka", new_fig=False)
# plt.subplot(235, sharex=ax, sharey=ax), imshow(bg, title="Background", new_fig=False)
# plt.subplot(236, sharex=ax, sharey=ax), imshow(not_bk, title="not_bk", new_fig=False)
# plt.suptitle(f'bg_RGB = {bg_color_rgb} | bg_hsv = {bg_color_hsv} | delta = {delta_bg}')
# plt.show(block=False)

# #Debemos recortar el frame actual para que solo se analice la parte que esta dentro del fondo
# imshow(mask_bk)

# ### Una vez obtenida el area de interés, cortamos la máscara de not_bk por
# ### las mismas dimensiones

# mask_not_bk_interes = mask_not_bk[y:y+h, x:x+h]
# imshow(mask_not_bk_interes)

# k = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
# mask_open = cv2.morphologyEx(mask_not_bk_interes, cv2.MORPH_OPEN, k)
# imshow(mask_open)
# mask_blur = cv2.GaussianBlur(mask_open, (5,5), 0)
# imshow(mask_blur)

# _, mask_blur_binary = cv2.threshold(mask_blur, 100,255, cv2.THRESH_BINARY)
# imshow(mask_blur_binary)


# #finalmente se pueden detectar los cuadrados:

# num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_blur_binary, connectivity=4)
# img_draw = cv2.merge((mask_not_bk_interes,mask_not_bk_interes,mask_not_bk_interes))
# #Obtenemos las componentes conectadas y aplicamos filtro por área
# # Filtra componentes por área
# for i in range(1, num_labels):  # Omite la etiqueta 0 (el fondo)
#     area = stats[i, cv2.CC_STAT_AREA]  # Área de la componente
#     # Obtén las coordenadas del bounding box
#     x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
#     if area >= 2000 or h>100 or area < 600:  # Filtra componentes grandes
#         # Dibuja el rectángulo en verde
#         cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     else:
#         # Dibuja el rectángulo en rojo para áreas pequeñas
#         cv2.rectangle(img_draw, (x, y), (x + w, y + h), (255, 0, 0), 2)

# imshow(img_draw, title= '4. Imagen Filtrada')


# # #Aplicamos apertura ya que debemos estirar el fondo verde

# # Quitamos áreas muy grandes que corresponden al borde de la patente
# num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_blur_binary, connectivity=8)

# filtered_mask_not_bk = mask_not_bk.copy()
# img_draw = cv2.merge((mask_not_bk,mask_not_bk,mask_not_bk))
# #Obtenemos las componentes conectadas y aplicamos filtro por área
# not_cuadrados = []
# # Filtra componentes por área
# for i in range(1, num_labels):  # Omite la etiqueta 0 (el fondo)
#     area = stats[i, cv2.CC_STAT_AREA]  # Área de la componente
#     if area >= 2000 or area < 600:  # Filtra componentes grandes
#         # Obtén las coordenadas del bounding box
#         x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
#         # Dibuja el rectángulo en verde
#         cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         filtered_mask_not_bk[y:y+h,x:x+w] = 0
#     else:
#         # Dibuja el rectángulo en rojo para áreas pequeñas
#         x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
#         cv2.rectangle(img_draw, (x, y), (x + w, y + h), (255, 0, 0), 2)
# imshow(filtered_mask_not_bk)
# imshow(img_draw, title= '4. Imagen Filtrada')




# # print(len(circulos))
# # print(numero_frame)
