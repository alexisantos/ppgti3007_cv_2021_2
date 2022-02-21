import cv2
import glob
import random
from imutils import encodings
import numpy as np
import os
import pickle

# inicializa o módulo LBP
fr = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)

# inicializa a lista de rótulos
labels = []

# loop nas imagens de entrada para treino
for (i, path) in enumerate(glob.glob('output/faces/*.txt')):
    # extrai o nome da pessoa a partir do nome do arquivo
    name = path[path.rfind("/") + 1:].replace(".txt", "")
    print("[INFO] treinando '{}'".format(name))

    # carrega os arquivos de face, realiza uma amostragem e inicializa a lista de faces
    sample = open(path).read().strip().split("\n")

    # amostra de 100 faces
    sample = random.sample(sample, min(len(sample), 100))
    faces = []
    # loop nas faces da amostra
    for face in sample:
        # decodifica a face de base64 e atualiza a lista        
        faces.append(encodings.base64_decode_image(face))
    # treina o detector de faces e atualiza a lista de rótulos    
    fr.train(faces, np.array([i] * len(faces)))
    labels.append(name)
# atualiza o reconhecedor de faces para inlcuir os rótulos


# cria o arquivo 
if not os.path.exists('./output/classifier/classifier.model'):
    f = open("./output/classifier/classifier.model", "w+")
    f.close()

fr.write("./output/classifier/classifier.model")

# salva os parâmetros do modelo
f = open("./output/classifier/fr.cpickle", "wb")
f.write(pickle.dumps(labels))
f.close()