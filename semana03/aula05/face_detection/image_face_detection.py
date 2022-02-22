import cv2

# inicializa o detector de face
fd = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')

# carrega imagem de teste
image = cv2.imread('data/selecao.jpg')
# converte para escala de cinza
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detecta faces no frame
# parametros:
    # scaleFactor: fator de redimensionamento a cada camada da piramide de imagens
    # minNeighbors: quantas janelas vizinhas devem ser consideradas para afirmar que o objeto foi detectado
    # minSize = tamanho minimo que a janela deve ter para ser considerada 
bbs = fd.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=9, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

# loop nos bounding boxes detectados
for (i, (x, y, w, h)) in enumerate(bbs):
    # extrai a face
    face = gray[y:y + h, x:x + w]
    # desenha o bb na tela        
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# mostra o resultado
cv2.imshow('Frame', image)
key = cv2.waitKey(0)
cv2.destroyAllWindows()