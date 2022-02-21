import imutils
import cv2
import pickle

# inicializa o detector de face
fd = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
# carrega o modelo que reconhece a face
labels = pickle.loads(open('./output/classifier/fr.cpickle', 'rb').read())
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('./output/classifier/classifier.model')

# determina o nível de confiança
recognizer.setThreshold(70)
# grab a reference to the webcam

# inicializa a webcam
camera = cv2.VideoCapture(0)

# loop nos frames do video
while True:
    # lê o frame atual
    (grabbed, frame) = camera.read()

    # se não puder ser lido, o video chegou ao final
    if not grabbed:
        break

    # redimensiona o frame
    frame = imutils.resize(frame, width=500)
    # converte para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecta faces no frame
    faceRects = fd.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=9, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

    # loop nos bounding boxes detectados
    for (i, (x, y, w, h)) in enumerate(faceRects):
        # extrai a face
        face = gray[y:y + h, x:x + w]
        
        # prediz de quem é a face
        (prediction, confidence) = recognizer.predict(face)
        if prediction == -1:
            prediction = 'Desconhecido'
        else:
            prediction = labels[prediction]            
        prediction = "{}: {:.2f}".format(prediction, confidence)
        # mostra o texto na imagem e desenha o retângulo ao redor da imagem
        cv2.putText(frame, prediction, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # mostra o frame e verifica se o usuário pressional alguma tecla    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # se a tecla q for pressionada, interrompe o loop
    if key == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()
