Este projeto consiste no pipeline completo para construir um modelo que reconhece faces em tempo real. Para demonstrar, vamos utilizar o feed da webcam para capturar e exibir o resultado do reconhecimento.

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

## Pipeline de reconhecimento facial
A partir disso, basta seguir com o pipeline que consiste em três passos, divididos em três arquivos que devem ser executados nessa ordem:

1. `collect_faces.py`: script que utiliza um modelo pré-treinado (cascades) para realizar a detecção facial e salvar faces a serem reconhecidas posteriormente. O uso do script deve ser da seguinte maneira:
```shell
python collect_faces.py --name NOME_DA_PESSOA
```
Deve-se inserir o nome da pessoa no lugar de `NOME_DA_PESSOA`. Ao executar o script, seu feed da webcam é aberto e se existirem faces na tela, o algoritmo deve detectar. Para iniciar a captura das faces a serem utilizadas no reconhecedor de faces, deve-se pressionar a tecla `c` no teclado (note que o bounding box mudar de cor). Para finalizar a captura de faces, a tecla `c` deve ser novamente pressionada (é ideal capturar, pelo menos, uns 5 segundos com posições variadas da face). Para finalizar o script, basta pressionar a tecla `q`.

Se desejar adicionar mais de uma pessoa à base de dados, execute o script novamente trocando o nome da pessoa no parâmetro de inicialização.

2. `train_recognizer.py`: script que vai construir um classificador com base nas imagens capturadas com o script anterior. As imagens capturadas ficaram salvas no diretório `output/faces` codificadas em *base64* e armazenadas no formato binário `pickle`. Para executar o script, basta executar:

```shell
python train_recognizer.py
```

Nesse passo, uma versão do algoritmo 1-NN juntamente com a codificação extraída utilizando *Local Binary Patterns* é usado para construir o modelo.

3. `recognize.py`: por último, esse script carrega o modelo construído no passo anterior e armazenado no diretório `output/classifier` é utilizado para realizar o reconhecimento em tempo real das faces detectadas na captura da webcam. Para executar o script, basta executar:

```shell
python recognizer.py
```