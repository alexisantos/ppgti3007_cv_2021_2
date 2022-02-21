import cv2
import argparse
import os
import base64
import utils
import pickle

# função auxiliar para detectar faces na imagem
def detect(fd, image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
	# detecta faces na imagem
	flags = cv2.CASCADE_SCALE_IMAGE
	bbs = fd.detectMultiScale(image, scaleFactor=scaleFactor,
		minNeighbors=minNeighbors, minSize=minSize, flags=flags)

	# retorna os bounding boxes encontrados
	return bbs

# verificação e inicialização de parâmetros
ap = argparse.ArgumentParser()
ap.add_argument('-n', '--name', help='nome da pessoa cujas faces vão ser coletadas.')
args = ap.parse_args()
if not args.name:
	args.name = 'person'

# inicializa o detector de faces haar cascade com o modelo pré-treinado
fd = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
# flag para indicar se estamos capturando imagens
capture_mode = False
# cor do bounding box
color = (0, 255, 0)

# inicializa a câmera
camera = cv2.VideoCapture(0)
# abre o arquivo onde vão ficar salvas as imagens no formato base64
output_path = 'output/faces'
if not os.path.exists(output_path):
	os.mkdir('output/')
	os.mkdir('output/faces/')
total = 0
# lista para armzenar as imagens em formato base64
faces_b64 = []

# loop nos frames do video capturado
while True:
	# obtém o frame atual
	(grabbed, frame) = camera.read()

	# se o frame não pode ser capturado, encerra o loop
	if not grabbed:
		break

	# obtem informações sobre as dimensões do frame para redimensionar
	(h,w,c) = frame.shape
    # largura desejada
	w_desired = 500
    # redimensiona o frame
	frame = cv2.resize(frame,(w_desired,int((h/w)*w_desired)),interpolation = cv2.INTER_AREA)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	bbs = detect(fd,gray, scaleFactor=1.1, minNeighbors=9, minSize=(30, 30))

	# garante que uma face é detectada, pelo menos
	if len(bbs) > 0:
		# ordena os bbs, mantendo somente o maior		
		(x, y, w, h) = max(bbs, key=lambda b:(b[2] * b[3]))

		# se estiver no modo de captura, salva as faces no arquivo
		if capture_mode:			
			# extrai a ROI (face)
			face = gray[y:y + h, x:x + w].copy(order="C")			
			# adiciona a face a lista de faces no formato base64 
			faces_b64.append(utils.img2b64(face, '.jpg'))
			total += 1

		# desenha o bb
		cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

	# show the frame and record if the user presses a key
	cv2.imshow('Frame', frame)
	key = cv2.waitKey(1) & 0xFF

	
	# se a tecla 'c' é pressionada, entra no modo captura
	if key == ord('c'):
		# se não estiver no modo captura, entra no modo captura		
		if not capture_mode:
			capture_mode = True
			# muda a cor do bb para indicar que está no modo captura
			color = (0, 0, 255)
		# se já está, sai do modo captura		
		else:
			capture_mode = False
			color = (0, 255, 0)

	# o loop encerra quando a tecla 'q' é pressionada	
	elif key == ord('q'):
		break


print(f'[INFO] {total} frames foram salvos')
with open(f'./output/faces/{args.name}.pickle', 'wb') as f:
    pickle.dump(faces_b64,f)
# libera a camera
camera.release()
cv2.destroyAllWindows()