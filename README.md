# Perceptron - Atividade

Implementacao do Perceptron (Pocket PLA) pra classificar digito 1 vs digito 5
no dataset disponibilizado.

## Como rodar

```
pip install numpy matplotlib jupyter
python perceptron.py
```

Ou abrir o notebook `perceptron.ipynb`.

## Arquivos

- `perceptron.ipynb` - notebook principal com a analise
- `perceptron.py` - mesma coisa como script
- `digits.train` / `digits.test` - datasets
- `fronteira_treino.png` / `fronteira_teste.png` - desafio
- `curva_aprendizado.png` - erro ao longo das iteracoes

## Resultado

| | E_in | E_out |
|-|-|-|
| PLA classico | ~0.6% | ~3.0% |
| Pocket PLA | ~0.3% | ~2.1% |
