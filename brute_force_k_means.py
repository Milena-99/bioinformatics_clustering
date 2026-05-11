"""
Brute-force K-means — iscrpno trazenje optimuma (referenca za Lloyd algoritam).

Pretrazuje sve podskupove od k tacaka iz podataka kao kandidate za centre
(C(n, k) kombinacija), bira onaj sa minimalnom distorzijom.

Slozenost: O(C(n,k) · n · k · d). Prakticno samo za male n.

Napomena: Lloyd algoritam koristi centroide (sredine klastera) koji ne moraju
biti tacke iz podataka — pa Lloyd moze postici i manju distorziju od ove
brute-force varijante. Ovo je u sustini k-medoids brute force.
"""

import random
from utils import euclidean_distance, genes, gene_names


class BruteForceKMeans:
    def __init__(self, k):
        self.k = k
        self.centers = []
        self.labels = []
        self.best_distortion = float('inf')

    '''Generator svih C(n, k) kombinacija indeksa (rekurzivno, bez itertools)'''
    def combinations(self, n, k, start=0):
        if k == 0:
            yield []
            return
        for i in range(start, n - k + 1):
            for rest in self.combinations(n, k - 1, i + 1):
                yield [i] + rest

    '''Distorzija: prosecno kvadratno rastojanje tacke do najblizeg centra'''
    def distortion(self, data, centers):
        total = 0.0
        for point in data:
            min_r = min(euclidean_distance(point, c) for c in centers)
            total += min_r ** 2
        return total / len(data)

    '''Dodela tacaka klasterima po najblizem centru'''
    def assign_clusters(self, data):
        self.labels = []
        for point in data:
            min_r = float('inf')
            best = 0
            for i, c in enumerate(self.centers):
                r = euclidean_distance(point, c)
                if r < min_r:
                    min_r = r
                    best = i
            self.labels.append(best)

    '''Iscrpna pretraga: isprobaj sve C(n, k) podskupove tacaka kao centre'''
    def fit(self, data):
        n = len(data)
        self.best_distortion = float('inf')
        self.centers = []

        n_tried = 0
        for indices in self.combinations(n, self.k):
            candidate_centers = [data[i][:] for i in indices]
            d = self.distortion(data, candidate_centers)
            n_tried += 1

            if d < self.best_distortion:
                self.best_distortion = d
                self.centers = candidate_centers

        self.assign_clusters(data)
        return n_tried


if __name__ == '__main__':
    print("=" * 60)
    print("Brute-force K-means — test na podacima genske ekspresije")
    print("=" * 60)

    bfkm = BruteForceKMeans(k=3)
    n_tried = bfkm.fit(genes)

    print(f"\nIsprobano kombinacija: {n_tried}  (C({len(genes)}, 3))")
    print(f"Najmanja distorzija: {bfkm.best_distortion:.4f}\n")

    for cluster_id in range(3):
        members = [gene_names[i] for i in range(len(genes)) if bfkm.labels[i] == cluster_id]
        print(f"  Klaster {cluster_id + 1}: {members}")

    print("\nNajbolji centri (tacke iz podataka, medoid varijanta):")
    for i, c in enumerate(bfkm.centers):
        print(f"  Centar {i + 1}: [{', '.join(f'{x:+.2f}' for x in c)}]")

    # Poredjenje sa Lloyd
    try:
        from k_means import KMeans
        print("\nPoredjenje sa Lloyd algoritmom (5 seed-ova):")
        for seed in [0, 1, 7, 42, 100]:
            random.seed(seed)
            km = KMeans(k=3)
            km.fit(genes)
            print(f"  seed={seed:>3}  Lloyd = {km.distortion(genes):.4f}   BFKM = {bfkm.best_distortion:.4f}")
        print("\n  (Lloyd moze biti nizi jer koristi centroide, ne tacke iz podataka.)")
    except ImportError:
        pass
