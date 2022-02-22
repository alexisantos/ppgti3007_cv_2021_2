import cv2
import base64
import numpy as np

def img2b64(image, ext):
    # cria um buffer 1-dim da imagem BGR (OpenCV)
    _, image_array = cv2.imencode(ext, image)
    # converte a imagem 1-dim para bytes
    image_bytes = image_array.tobytes()
    # codifica a imagem 1-dim para base64
    image_b64 = base64.b64encode(image_bytes)
    # retorna a imagem base64
    return image_b64

def b642img(image_b64):
    # decodifica a imagem do formato base64 para bytes
    image_bytes = base64.b64decode(image_b64)
    # transforma a imagem em bytes para um array 1-dim
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    # decodifica a imagem em bytes para um array BGR (OpenCV)
    image = cv2.imdecode(image_array, flags=cv2.IMREAD_GRAYSCALE)
    # retorna a imagem OpenCV
    return image