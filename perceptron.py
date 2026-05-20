# Perceptron - atividade de aprendizado de maquina
# usei o Pocket PLA pq vi na aula que se os dados nao sao separaveis
# o PLA normal nao para

import numpy as np
import matplotlib.pyplot as plt


# carrega os dados do arquivo
# o formato e: digito, intensidade, simetria
def carregar(path):
    dados = np.loadtxt(path)
    y = dados[:, 0]
    X = dados[:, 1:]

    # filtro so as classes 1 e 5 (igual o exemplo do livro)
    idx = (y == 1) | (y == 5)
    X = X[idx]
    y = y[idx]

    # +1 pra digito 1, -1 pra digito 5
    y = np.where(y == 1, 1, -1)

    # coloco uma coluna de 1's no comeco pro bias
    uns = np.ones((X.shape[0], 1))
    X = np.hstack([uns, X])

    return X, y


def sinal(z):
    s = np.sign(z)
    s[s == 0] = 1
    return s


def erro_in(w, X, y):
    pred = sinal(X @ w)
    return np.mean(pred != y)


# PLA classico, so pra comparar com o pocket
def pla(X, y, max_iter=5000):
    np.random.seed(0)
    w = np.zeros(X.shape[1])
    historico = []
    for i in range(max_iter):
        pred = sinal(X @ w)
        errados = np.where(pred != y)[0]
        historico.append(len(errados) / len(y))
        if len(errados) == 0:
            break
        # pega um errado aleatorio
        j = np.random.choice(errados)
        w = w + y[j] * X[j]
    return w, historico


# Pocket PLA: igual ao PLA mas guarda o melhor w
def pocket(X, y, max_iter=5000):
    np.random.seed(0)
    w = np.zeros(X.shape[1])
    melhor_w = w.copy()
    melhor_erro = erro_in(w, X, y)
    hist = []

    for i in range(max_iter):
        hist.append(melhor_erro)

        pred = sinal(X @ w)
        errados = np.where(pred != y)[0]
        if len(errados) == 0:
            break

        j = np.random.choice(errados)
        w = w + y[j] * X[j]

        # checa se esse w novo e melhor
        e = erro_in(w, X, y)
        if e < melhor_erro:
            melhor_erro = e
            melhor_w = w.copy()

    return melhor_w, melhor_erro, hist


# plot da fronteira (desafio)
# como w0 + w1*x + w2*y = 0, dai isolando y da pra desenhar
def plot_fronteira(w, X, y, titulo, nome_arquivo):
    plt.figure(figsize=(7, 6))
    plt.scatter(X[y == 1, 1], X[y == 1, 2], c='blue', marker='o', s=15, alpha=0.6, label='digito 1')
    plt.scatter(X[y == -1, 1], X[y == -1, 2], c='red', marker='x', s=18, alpha=0.6, label='digito 5')

    x_eixo = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    y_eixo = -(w[0] + w[1] * x_eixo) / w[2]
    plt.plot(x_eixo, y_eixo, 'k-', label='fronteira')

    plt.xlabel('intensidade')
    plt.ylabel('simetria')
    plt.title(titulo)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(X[:, 2].min() - 0.5, X[:, 2].max() + 0.5)
    plt.savefig(nome_arquivo, dpi=110)
    plt.close()


if __name__ == '__main__':
    X_tr, y_tr = carregar('digits.train')
    X_te, y_te = carregar('digits.test')

    print('amostras treino:', len(y_tr))
    print('amostras teste :', len(y_te))

    # primeiro o PLA normal
    w1, h1 = pla(X_tr, y_tr, max_iter=5000)
    print('\nPLA classico:')
    print('  E_in =', round(erro_in(w1, X_tr, y_tr), 4))
    print('  E_out=', round(erro_in(w1, X_te, y_te), 4))

    # agora o pocket
    w2, ein, hist = pocket(X_tr, y_tr, max_iter=5000)
    eout = erro_in(w2, X_te, y_te)
    print('\nPocket PLA:')
    print('  pesos:', w2)
    print('  E_in =', round(ein, 4))
    print('  E_out=', round(eout, 4))

    # plots
    plot_fronteira(w2, X_tr, y_tr,
                   'Pocket PLA - treino (E_in = %.2f%%)' % (ein * 100),
                   'fronteira_treino.png')
    plot_fronteira(w2, X_te, y_te,
                   'Pocket PLA - teste (E_out = %.2f%%)' % (eout * 100),
                   'fronteira_teste.png')

    # curva do erro
    plt.figure(figsize=(8, 4))
    plt.plot(h1, color='gray', alpha=0.7, label='PLA classico')
    plt.plot(hist, color='green', label='Pocket (melhor erro)')
    plt.xlabel('iteracao')
    plt.ylabel('E_in')
    plt.title('Erro ao longo das iteracoes')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('curva_aprendizado.png', dpi=110)
    plt.close()

    print('\nplots salvos.')
