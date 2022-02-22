import cv2

# inicializa o detector de face
fd = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')

# inicializa a webcam
camera = cv2.VideoCapture(0)

# loop nos frames do video
while True:
    # lê o frame atual
    (grabbed, frame) = camera.read()    

    # se não puder ser lido, o video chegou ao final
    if not grabbed:
        break

    # obtem as dimensoes originais do frame de captura
    (h,w,c) = frame.shape
    # largura desejada
    w_desired = 500
    # redimensiona o frame
    frame = cv2.resize(frame,(w_desired,int((h/w)*w_desired)),interpolation = cv2.INTER_AREA)    
    # converte para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecta faces no frame
    # parametros:
        # scaleFactor: fator de redimensionamento a cada camada da pirâmide de imagens
        # minNeighbors: quantas janelas vizinhas devem ser consideradas para afirmar que o objeto foi detectado
        # minSize = tamanho minimo que a janela deve ter para ser considerada 
    bbs = fd.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=9, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # loop nos bounding boxes detectados
    for (i, (x, y, w, h)) in enumerate(bbs):
        # extrai a face
        face = gray[y:y + h, x:x + w]
        # desenha o bb na tela        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # mostra o frame
    cv2.imshow('Frame', frame)
    # verifica se alguma tecla foi pressionada
    key = cv2.waitKey(1) & 0xFF
    # se a tecla q for pressionada, interrompe o loop
    if key == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()