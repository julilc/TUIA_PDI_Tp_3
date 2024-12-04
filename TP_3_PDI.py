from rqst import *

def area_of_interest(frame_bg: np.array, stop: bool = False)->np.array:
    # Tomo una muestra
    frame_hsv = cv2.cvtColor(frame_bg, cv2.COLOR_RGB2HSV)
    bg_crop_RGB = frame_bg[30:200, 30:200, :]                     # Cortar una porcion correspondiente al fondo.
    #imshow(bg_crop_RGB, title="Frame 1 - Crop fondo verde")
    img_pixels = bg_crop_RGB.reshape(-1,3)
    colours, counts = np.unique(img_pixels, axis = 0, return_counts= True)
    n_colours = colours.shape[0]
    n_colours
    colours[0]

    bg_color_rgb = colours[4]
    bg_color_rgb = np.reshape(bg_color_rgb, (1,1,-1))
    bg_color_hsv = cv2.cvtColor(bg_color_rgb, cv2.COLOR_RGB2HSV)
    DELTA_BG = np.array([10.0, 150.0, 150.0])
    #Define limite inferior y superior en rango de verde.
    lower_limmit = np.clip(bg_color_hsv - DELTA_BG, 0, 255)
    upper_limmit = np.clip(bg_color_hsv + DELTA_BG, 0, 255)

    #Obtiene mascara de bk y no bk
    mask_bk = cv2.inRange(frame_hsv, lower_limmit, upper_limmit)
    mask_not_bk = cv2.bitwise_not(mask_bk)

    #obtiene bk y no bk del frame
    bg = cv2.bitwise_and(frame_bg, frame_bg, mask=mask_bk)
    not_bk = cv2.bitwise_and(frame_bg, frame_bg, mask=mask_not_bk)
    
    ### Análisis bk para area de interés ##
    
    #Obtiene componentes conectadas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bk, connectivity=8)
    
    #Ordena por area de mayor a menor
    # Extraer las áreas y sus índices
    areas = stats[:, cv2.CC_STAT_AREA]  # Áreas de cada componente
    sorted_indices = np.argsort(areas)[::-1]  # Ordenar en orden descendente
    
    # Reordenar stats y centroids según las áreas
    sorted_stats = stats[sorted_indices]
    sorted_centroids = centroids[sorted_indices]
    
    # ignoramos el fondo.
    sorted_stats = sorted_stats[1:]
    sorted_centroids = sorted_centroids[1:]

    #Obtiene el area verde de la imagen.
    fondo = sorted_indices[0]

    # Obtener las coordenadas del bounding box para esta componente
    x_area_interes, y_area_interes, w_area_interes, h_area_interes = stats[fondo, cv2.CC_STAT_LEFT], stats[fondo, cv2.CC_STAT_TOP], stats[fondo, cv2.CC_STAT_WIDTH], stats[fondo, cv2.CC_STAT_HEIGHT]

    #Recorta el frame por area de interés
    area_interes = frame_bg[y_area_interes:y_area_interes+h_area_interes, x_area_interes:x_area_interes+w_area_interes]
    #imshow(area_interes)
    
    ##### A raíz de ese Frame, detecta los cuadrados.
    
    #Solo los analiza si es el frame donde ya pararon de girar
    if stop:
        #Corta la máscara de not_bk por las mismas dimensiones que el area de interes

        mask_not_bk_interes = mask_not_bk[y_area_interes:y_area_interes+h_area_interes, x_area_interes:x_area_interes+h_area_interes]
        #imshow(mask_not_bk_interes)

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
        mask_open = cv2.morphologyEx(mask_not_bk_interes, cv2.MORPH_OPEN, k)
        #imshow(mask_open)
        mask_blur = cv2.GaussianBlur(mask_open, (5,5), 0)
        #imshow(mask_blur)

        _, mask_blur_binary = cv2.threshold(mask_blur, 100,255, cv2.THRESH_BINARY)
        #imshow(mask_blur_binary)


        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_blur_binary, connectivity=4)
        img_draw = area_interes.copy()
        #Obtenemos las componentes conectadas y aplicamos filtro por área
        # Filtra componentes por área
        c= 0
        for i in range(1, num_labels):  # Omite la etiqueta 0 (el fondo)
            area = stats[i, cv2.CC_STAT_AREA]  # Área de la componente
            # Obtén las coordenadas del bounding box
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                
            if area >= 2000 or h>100 or area < 600:  # Filtra componentes grandes
                # Dibuja el rectángulo en verde
                continue
            else:
                c += 1
                # Dibuja el rectángulo en rojo para cuadrados
                cv2.rectangle(img_draw, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cuadrado = area_interes[y:y+h, x:x+w]
                _, frame_bin = cv2.threshold(cv2.cvtColor(cuadrado, cv2.COLOR_RGB2GRAY), thresh=80, maxval=255, type=cv2.THRESH_BINARY)
                cuadrado, circulos = find_and_draw_circles(frame_bin,dp_ = 1.2, minD=7,minR=2,p1=2,p2= 8,maxR=8)
                label_text= f'Cuadrado {c}'
                cv2.putText(img_draw, label_text, (x-70, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                label_text = f'puntos {len(circulos)}'
                cv2.putText(img_draw, label_text, (x-70, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                
        
        imshow(img_draw, title= '4. Imagen Filtrada', blocking= True)
        return img_draw, (x_area_interes, y_area_interes, w_area_interes, h_area_interes)
    
    
    return area_interes, (x_area_interes, y_area_interes, w_area_interes, h_area_interes)



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

    # Obtener frame actual y área de interés
    frame_actual, coords = area_of_interest(cv2.cvtColor(frame_actual, cv2.COLOR_BGR2RGB))
    frame_actual = cv2.resize(frame_actual, None, fx=escala, fy=escala, interpolation=cv2.INTER_LINEAR)
    x, y, w, h = coords

    frame_number = 0

    while cap.isOpened():
        # Leer el siguiente frame
        ret, frame_siguiente = cap.read()
        frame_siguiente = cv2.cvtColor(frame_siguiente, cv2.COLOR_BGR2RGB)
        if not ret:
            print("Fin del video.")
            break

        # Recortar el frame siguiente al área de interés
        frame_siguiente = frame_siguiente[y:y+h, x:x+w]

        # Redimensionar el frame siguiente para que coincida con el actual
        frame_siguiente = cv2.resize(frame_siguiente, (frame_actual.shape[1], frame_actual.shape[0]), interpolation=cv2.INTER_LINEAR)

      
        # Calcular la diferencia entre frames
        diferencia = cv2.absdiff(frame_actual, frame_siguiente)
        # Usamos un umbral para resaltar las diferencias
        _, diferencia_bin = cv2.threshold(diferencia, 30, 255, cv2.THRESH_BINARY)

        # Vemos cuantos pixeles se mueven para retener los que son 0
        num_pixeles_movimiento = np.sum(diferencia_bin > 0)
        paro = False
        
        if num_pixeles_movimiento == 0 and frame_number > 50:
            paro = True
            #imshow(frame_actual, blocking=True)
            area_of_interest(frame_actual, stop = True)
        
        
        if num_pixeles_movimiento != 0 and paro:
            return
            
        frame_number+=1
        
        
        
        frame_actual = frame_siguiente
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("Detenido por el usuario.")
            break
    cap.release()
    cv2.destroyAllWindows()

for i in range(1,2):
    video_path = f"data/tirada_{i}.mp4"

    frame, numero_frame = detectar_movimiento(video_path, umbral_movimiento=0, escala=0.5)

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
