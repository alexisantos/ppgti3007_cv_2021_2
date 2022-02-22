Este projeto mostra uma forma especial de deteção de objetos, o detector de faces, que pode ser construídos utilizando os mesmos princípios da deteção de objetos, mas fazendo uso do algoritmo de Viola-Jones (chamado também de detector haar cascade). 

Temos dois exemplos muito parecidos:
- [`image_face_detection.py`](./image_face_detection.py): detecta faces em uma imagem estática.
- [`video_face_detection.py`](./video_face_detection.py): detecta faces em tempo real pela captura de video da webcam.

## Preparação do ambiente

Recomenda-se antes de iniciar a execução dos scripts, criar um ambiente virtual python para isolar a instalação dos pacotes. Para isso, basta digitar no terminal (considerando sistemas linux ou mac):

```shell
python3 -m venv NOME_DO_AMBIENTE
```
em que `NOME_DO_AMBIENTE` é o nome dado para o seu ambiente virtual. Após isso, você pode inicializar o seu ambiente virtual executando:

```shell
source NOME_DO_AMBIENTE/bin/activate
```
Por fim, os pacotes utilizados podem ser instalados por meio do `pip` e o arquivo `requirements(x86|arm64).txt` (lembrando de atualizar o `pip` primeiro):
```shell
pip install --upgrade pip
pip install -r requirements(x86|arm64).txt
```
Um detalhe importante é que podem existir dois arquivos diferentes dentro do repositório para a lista de pacotes: `requirements_x86.txt` ou `requirements_arm64.txt`. O primeiro deve ser utilizado em computadores com chips Intel/AMD x86 e o segundo com chips Apple Silicon.

Para executar o script, basta fazer:

- para imagens:
```shell
python image_face_detection.py
```
- para videos:
```shell
python video_face_detection.py
```