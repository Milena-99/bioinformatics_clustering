"""
Meki K-means (Soft K-means) - EM pristup sa matricom odgovornosti.

Za razliku od Lloyd algoritma koji kruto dodeljuje svaku tacku jednom klasteru,
meki K-means dodeljuje odgovornosti (responsibilities) - koliko svaki klaster
"privlaci" svaku tacku.

E-korak: izracunaj matricu odgovornosti R_{ij} = exp(-beta * d_{ij}) / norm
M-korak: azuriraj centre kao ponderisane centre gravitacije
"""

import math
import random
from utils import euclidean_distance, genes, gene_names


class SoftKMeans:
    def __init__(self, k, beta=2.0, max_iter=100, tolerance=1e-4):
        self.k = k
        self.beta = beta
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.centers = []
        self.responsibility_matrix = []   # k x n matrica (HiddenMatrix)

    '''Inicijalizacija centara (k-means++)'''
    def init_centers(self, data):
        self.centers = [data[random.randint(0, len(data) - 1)][:]]

        for _ in range(1, self.k):
            distances = []
            for point in data:
                min_r = min(euclidean_distance(point, c) for c in self.centers)
                distances.append(min_r ** 2)

            total = sum(distances)
            r = random.random() * total
            cumulative = 0
            for i, d in enumerate(distances):
                cumulative += d
                if cumulative >= r:
                    self.centers.append(data[i][:])
                    break

    '''E-korak: izracunaj matricu odgovornosti R (klaster x tacka)'''
    def e_step(self, data):
        n = len(data)
        self.responsibility_matrix = [[0.0] * n for _ in range(self.k)]

        for j in range(n):
            exponents = []
            for i in range(self.k):
                d = euclidean_distance(data[j], self.centers[i])
                exponents.append(-self.beta * d)

            # Numericka stabilnost: oduzmi maksimum pre exp()
            max_val = max(exponents)
            exp_vals = [math.exp(e - max_val) for e in exponents]
            total = sum(exp_vals)

            for i in range(self.k):
                self.responsibility_matrix[i][j] = exp_vals[i] / total

    '''M-korak: azuriraj centre kao ponderisane centre gravitacije'''
    def m_step(self, data):
        n = len(data)
        m = len(data[0])
        new_centers = []

        for i in range(self.k):
            sum_weights = sum(self.responsibility_matrix[i][j] for j in range(n))

            if sum_weights == 0:
                new_centers.append(self.centers[i][:])
                continue

            new_center = []
            for dim in range(m):
                weighted_sum = sum(
                    self.responsibility_matrix[i][j] * data[j][dim] for j in range(n)
                )
                new_center.append(weighted_sum / sum_weights)
            new_centers.append(new_center)

        return new_centers

    '''Pokretanje EM algoritma'''
    def fit(self, data):
        self.init_centers(data)

        for iteration in range(self.max_iter):
            self.e_step(data)

            old_centers = [c[:] for c in self.centers]
            self.centers = self.m_step(data)

            shift = max(euclidean_distance(old_centers[i], self.centers[i])
                        for i in range(self.k))

            if shift < self.tolerance:
                return iteration + 1

        return self.max_iter

    '''Krute labele (klaster sa najvecom odgovornoscu za svaku tacku)'''
    def hard_labels(self):
        n = len(self.responsibility_matrix[0])
        labels = []
        for j in range(n):
            max_resp = -1
            best = 0
            for i in range(self.k):
                if self.responsibility_matrix[i][j] > max_resp:
                    max_resp = self.responsibility_matrix[i][j]
                    best = i
            labels.append(best)
        return labels

    '''Odgovornosti svih klastera za datu tacku'''
    def responsibilities(self, point_idx):
        return [self.responsibility_matrix[i][point_idx] for i in range(self.k)]


if __name__ == '__main__':
    print("=" * 60)
    print("Meki K-means (EM) - test na podacima genske ekspresije")
    print("=" * 60)

    random.seed(42)
    mkm = SoftKMeans(k=3, beta=2.0)
    n_iter = mkm.fit(genes)

    print(f"\nKonvergirao nakon {n_iter} iteracija (beta=2.0)\n")

    print("Matrica odgovornosti:")
    print(f"{'Gen':<10} {'Kl.1':>8} {'Kl.2':>8} {'Kl.3':>8}  -> Kruti label")
    print("-" * 55)

    hard = mkm.hard_labels()
    for j in range(len(genes)):
        resp = mkm.responsibilities(j)
        print(f"{gene_names[j]:<10} {resp[0]:>8.4f} {resp[1]:>8.4f} {resp[2]:>8.4f}  -> Klaster {hard[j] + 1}")

    print("\nUticaj parametra beta na krutost dodele (1.0 = potpuno kruto):")
    for beta in [0.5, 1.0, 2.0, 5.0, 10.0]:
        random.seed(42)
        mkm_t = SoftKMeans(k=3, beta=beta)
        mkm_t.fit(genes)

        mean_max = 0
        for j in range(len(genes)):
            resp = mkm_t.responsibilities(j)
            mean_max += max(resp)
        mean_max /= len(genes)

        print(f"  beta={beta:<5.1f} -> prosecna maks. odgovornost: {mean_max:.4f}")
