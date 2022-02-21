import cv2
import glob
import os
import pickle
import utils
import numpy as np

# inicializa o módulo LBP
fr = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)

# inicializa a lista de rótulos
labels_names = []
labels = []
faces = []
# loop nas imagens de entrada para treino
for (i, path) in enumerate(glob.glob('./output/faces/*.pickle')):
    # extrai o nome da pessoa a partir do nome do arquivo
    name = path[path.rfind("/") + 1:].replace(".pickle", "")
    print(f"[INFO] coletando '{name}'")
    # carrega os arquivos de face, realiza uma amostragem e inicializa a lista de faces
    with open(path, 'rb') as file:
        data = file.read()
        images_b64 = pickle.loads(data)
    
    # loop nas faces da amostra
    for face in images_b64:        
        # decodifica a face de base64 e atualiza a lista
        faces.append(utils.b642img(face))
        labels.append(i)
    # treina o detector de faces e atualiza a lista de rótulos    
    labels_names.append(name)
print('[INFO] Treinando...')
fr.train(faces, np.array(labels))


# cria o diretório onde vai ficar armazenado o arquivo com o modelo treinado
if not os.path.exists('./output/classifier/'):
    os.mkdir('./output/classifier/')
# armazena o modelo
with open('./output/classifier/model', 'wb') as file:
    fr.write('./output/classifier/model')
with open('./output/classifier/labels', 'wb') as file:
    pickle.dump(labels_names, file)