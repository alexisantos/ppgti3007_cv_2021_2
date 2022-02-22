# Semana 03

A terceira semana da disciplina é destinada à apresentação dos conceitos de detecção e reconhecimento de objetos. Nessa semana, não teremos encontros síncronos, ficando o conteúdo ministrado por meio de notebooks no Google Colab.

*Conteúdo da semana*:

[Assíncrono](aula05/)
- Detecção de objetos
	- Janelas deslizantes
	- Preparação de dados
	- Treino
	- Non-maxima suppression
- Detecção de faces
- Reconhecimento facial

O conteúdo está dividido em três partes:
- [Detecção de objetos](aula05/object_detection/): serão mostrados por meio de um notebook Colab os princípios básicos para se construir um detector de objetos simples.
- [Detecção facial](aula05/face_detection/): como uma forma especial de deteção de objetos é mostrado como um detector de faces pode ser construídos utilizando os mesmos princípios do tópico anterior, mas fazendo uso do algoritmo de Viola-Jones (chamado também de detector haar cascade). Nesse item são mostrados exemplos tanto no contexto de uma imagem estática como no de vídeo em tempo real.
- [Reconhecimento facial](aula05/face_recognition/): como último passo, um pipeline de reconhecimento facial é construído (imagem e vídeo). O reconhecimento de uma face (objeto) consiste em identificar a que pessoa aquela face pertence. Então, por meio de três passos, vai ser possível construir um classificador simples de faces.