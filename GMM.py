import numpy as np

class GMM():
    def __init__(self, n_components, tol, means, covariances, weights):
        self.n_components = n_components
        self.tol = tol
        self.means = means
        self.covariances = covariances
        self.weights = weights

    def fit(self, X):
        print(f"Inicializando fit com parametros: \nComponentes: {self.n_components}\nTolerancia: {self.tol}\nMedias: {self.means}\nCovariancias: {self.covariances}\nPesos: {self.weights}")
        print("Treinando...")

        d = len(X[0])
        normais = np.zeros((len(X), self.n_components))
        responsabilidades = np.zeros((len(X), self.n_components))
        max_iter = 100
        log_verossimilhanca_antiga = -np.inf

        for iteracao in range(1, max_iter + 1):
            N = np.zeros(self.n_components)
            log_verossimilhanca_total = 0

            for i in range(len(X)):
                for j in range(self.n_components):
                    normais[i, j] = (1 / (((2*np.pi)**(d/2)) * np.sqrt(np.linalg.det(self.covariances[j])))) * np.exp(-0.5 * (X[i] - self.means[j]) @ np.linalg.inv(self.covariances[j]) @ (X[i] - self.means[j]))

            for i in range(len(X)):
                soma_normais = 0
                for j in range(self.n_components):
                    soma_normais += self.weights[j] * normais[i, j]

                log_verossimilhanca_total += np.log(soma_normais + 1e-10)
                
                for k in range(self.n_components):
                    responsabilidades[i, k] = (self.weights[k] * normais[i, k]) / soma_normais
            
            if abs(log_verossimilhanca_total - log_verossimilhanca_antiga) < self.tol:
                print(f"Convergiu na iteração {iteracao}")
                break
            log_verossimilhanca_antiga = log_verossimilhanca_total
            
            for k in range(self.n_components):
                for r in range(len(X)):
                    N[k] += responsabilidades[r, k]
                            
                self.weights[k] = N[k] / len(X)

                soma_ponderada = np.zeros(d)
                for i in range(len(X)):
                    soma_ponderada += responsabilidades[i, k] * X[i]
                self.means[k] = soma_ponderada / N[k]

                soma_covariancias = np.zeros((d, d))
                for i in range(len(X)):
                    erro = X[i] - self.means[k]
                    erro_coluna = erro.reshape((d, 1))
                    erro_linha = erro.reshape((1, d))

                    matriz_erro = erro_coluna @ erro_linha

                    soma_covariancias += responsabilidades[i, k] * matriz_erro
                self.covariances[k] = soma_covariancias / N[k]
            print("=" * 30)
            print(f"Parametros Iter({iteracao}): \n Medias: {self.means}\n Covariancias: {self.covariances}\n Pesos: {self.weights}")
        
        print("=" * 15 + "CONVERGIU" + "=" * 15)
        print(f"Modelo convergiu na iter {iteracao}, com parametros atualizados: \n Medias: {self.means}\n Covariancias: {self.covariances}\n Pesos: {self.weights}")
        return True
    
    def predict(self, X):
        probs = self.predict_proba(X)
        componentes_maior_r = []
        for x in range(len(X)):
            maior_prob = -1
            indice = -1
            for k in range(self.n_components):
                if probs[x, k] > maior_prob:
                    maior_prob = probs[x, k]
                    indice = k
            componentes_maior_r.append(indice)

        return componentes_maior_r
    
    def predict_proba(self, X):
        d = len(X[0])
        normais = np.zeros((len(X), self.n_components))
        responsabilidades = np.zeros((len(X), self.n_components))

        for i in range(len(X)):
            for j in range(self.n_components):
                normais[i, j] = (1 / (((2*np.pi)**(d/2)) * np.sqrt(np.linalg.det(self.covariances[j])))) * np.exp(-0.5 * (X[i] - self.means[j]) @ np.linalg.inv(self.covariances[j]) @ (X[i] - self.means[j]))

        for i in range(len(X)):
            soma_normais = 0
            for j in range(self.n_components):
                soma_normais += self.weights[j] * normais[i, j]

            for k in range(self.n_components):
                responsabilidades[i, k] = (self.weights[k] * normais[i, k]) / soma_normais
        
        return responsabilidades