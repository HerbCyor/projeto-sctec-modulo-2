# Pipeline Preditivo MNIST

## Problema que resolve

Classificação automática de dígitos manuscritos (0–9) a partir de imagens de 28x28 pixels em
escala de cinza (dataset `mnist_784`). O projeto compara três abordagens de classificação
multiclasse, mede o desempenho de cada uma no conjunto de teste, e testa a robustez dos
modelos contra dígitos nunca vistos no treino (class masking + inferência OOD) e contra
imagens de dígitos manuscritos pelo próprio autor do projeto.

## Técnicas e tecnologias

- **Python**, **scikit-learn** (KNN, Random Forest, métricas, split estratificado),
  **TensorFlow/Keras** (MLP), **matplotlib** / **seaborn** (visualizações), **PIL** / **OpenCV**
  / **scipy.ndimage** (pré-processamento das imagens manuscritas).
- **Modelos comparados:** KNN (`k=3`, `weights='distance'`), Random Forest (`n_estimators=300`,
  `max_depth=20`, treinada em pixels crus), MLP (camadas `[256, 128]`, `learning_rate=1e-3`,
  15 épocas).

Pipeline: **EDA → pré-processamento (split estratificado + normalização) → treino dos 3
modelos → avaliação comparativa no teste → testes de robustez (class masking, OOD, dígitos
manuscritos próprios)**.

## Como executar

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
pip install -r requirements.txt
jupyter notebook notebook_mnist.ipynb
```

Execute as células em ordem, de cima para baixo. Nenhum caminho absoluto é usado — o MNIST é
baixado sob demanda pelo `fetch_openml`, e as imagens do Desafio C são lidas de
`data/meus_digitos/` (caminho relativo).

## Resultados (resumo)

Métricas no conjunto de teste (14000 amostras, nunca usadas em treino/ajuste):

| Modelo | Acurácia | F1 (ponderado) | Tempo de treino |
|--------|----------|-----------------|------------------|
| KNN | 0,9721 | 0,9721 | instantâneo (aprendizado preguiçoso; custo está na predição) |
| Random Forest | 0,9660 | 0,9660 | 6,6 s |
| MLP | 0,9770 | 0,9770 | 10,1 s |

Nos 10 dígitos manuscritos pelo autor (`data/meus_digitos/`), o melhor modelo (MLP) acertou
**10/10**.

## Melhorias futuras

- Usar uma **CNN** (rede neural convolucional), feita especificamente para imagens — tende a
  superar a MLP simples usada aqui nesse tipo de tarefa.
- **Data augmentation**: gerar variações artificiais dos dígitos de treino (pequenas rotações,
  deslocamentos, mudanças de espessura do traço) para melhorar a robustez do modelo.
- **Early stopping** no treino da MLP: a curva de loss mostra que a validação para de melhorar
  por volta da época 7, então rodar as 15 épocas completas já introduz algum overfitting.
- Coletar mais dígitos manuscritos próprios — 10 imagens (uma por classe) é uma amostra pequena
  para um teste estatisticamente confiável.
- Calibrar as probabilidades do modelo ou adicionar uma forma explícita de rejeição ("não sei"),
  para mitigar o problema de falsa confiança (overconfidence) observado no teste OOD.

## Vídeo de demonstração

https://drive.google.com/file/d/1oUzsbyXmvfeQrqu-4Bb85PEfSB0y1ZSF/view?usp=sharing
