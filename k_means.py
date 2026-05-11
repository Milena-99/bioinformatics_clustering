"""
Lloyd algoritam (K-means) — iterativno klasterovanje.

Minimizuje kvadratnu gresku distorzije. Heuristika koja moze zaglaviti
u lokalnom minimumu. Inicijalizacija centara: k-means++.

Algoritam ponavlja dva koraka:
  1. Centri -> Klasteri  (dodeli svaku tacku najblizem centru)
  2. Klasteri -> Centri  (azuriraj centre kao centre gravitacije)
"""

import random
from utils import euclidean_distance, centroid, genes, gene_names, print_clusters_by_labels


class KMeans:
    def __init__(self, k, max_iter=100, tolerance=1e-6):
        self.k = k
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.centers = []
        self.labels = []

    '''k-means++ inicijalizacija centara — bira udaljene tacke sa verovatnocom ~ d^2'''
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

    '''Korak "Centri -> Klasteri": dodeli svaku tacku najblizem centru'''
    def assign_clusters(self, data):
        self.labels = []
        for point in data:
            min_r = float('inf')
            closest = 0
            for i, c in enumerate(self.centers):
                r = euclidean_distance(point, c)
                if r < min_r:
                    min_r = r
                    closest = i
            self.labels.append(closest)

    '''Korak "Klasteri -> Centri": azuriraj centre kao centre gravitacije'''
    def update_centers(self, data):
        new_centers = []
        for i in range(self.k):
            members = [data[j] for j in range(len(data)) if self.labels[j] == i]
            if members:
                new_centers.append(centroid(members))
            else:
                new_centers.append(self.centers[i][:])
        return new_centers

    '''Kvadratna greska distorzije'''
    def distortion(self, data):
        total = 0
        for j, point in enumerate(data):
            r = euclidean_distance(point, self.centers[self.labels[j]])
            total += r ** 2
        return total / len(data)

    '''Pokretanje Lloyd algoritma'''
    def fit(self, data):
        self.init_centers(data)

        for iteration in range(self.max_iter):
            self.assign_clusters(data)
            new_centers = self.update_centers(data)

            shift = max(euclidean_distance(self.centers[i], new_centers[i])
                        for i in range(self.k))
            self.centers = new_centers

            if shift < self.tolerance:
                return iteration + 1

        return self.max_iter


if __name__ == '__main__':
    print("=" * 60)
    print("Lloyd algoritam (K-means) — test na podacima genske ekspresije")
    print("=" * 60)

    random.seed(42)
    km = KMeans(k=3)
    n_iter = km.fit(genes)

    print(f"\nKonvergirao nakon {n_iter} iteracija")
    print(f"Distorzija: {km.distortion(genes):.4f}\n")

    print_clusters_by_labels(km.labels, gene_names)

    print("\nCentri klastera (centri gravitacije, mogu biti van podataka):")
    for i, c in enumerate(km.centers):
        print(f"  Centar {i + 1}: [{', '.join(f'{x:+.2f}' for x in c)}]")

    # Stabilnost preko vise random seed-ova
    print("\nStabilnost preko 5 razlicitih seed-ova:")
    for seed in [0, 1, 7, 42, 100]:
        random.seed(seed)
        km_s = KMeans(k=3)
        km_s.fit(genes)
        print(f"  seed={seed:>3}  distorzija = {km_s.distortion(genes):.4f}")
