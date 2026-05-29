"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

Density-based algoritam koji pronalazi klastere proizvoljnog oblika i identifikuje
sum (outlier-e). Broj klastera nije zadat - proizilazi iz strukture podataka i
parametara eps (radijus okoline) i min_pts (minimalan broj suseda).

Tipovi tacaka:
  - Jezgrene (core):     imaju bar min_pts u eps-okolini
  - Granicne (border):   nisu jezgrene ali pripadaju eps-okolini neke jezgrene
  - Sum (noise):         label = -1
"""

from utils import euclidean_distance, genes, gene_names


'''Oznake stanja tacke: 0 = nije razmatrana, -1 = sum, >0 = ID klastera'''
class DBSCAN:
    def __init__(self, eps, min_pts):
        self.eps = eps
        self.min_pts = min_pts
        self.labels = []

    '''Pronalazenje suseda tacke u eps-okolini'''
    def eps_neighbors(self, idx, data):
        neighbors = []
        for j in range(len(data)):
            if euclidean_distance(data[idx], data[j]) <= self.eps:
                neighbors.append(j)
        return neighbors

    '''Sirenje klastera C iz seme P kroz gusto-povezane tacke (BFS)'''
    def expand_cluster(self, data, p, neighbors, c):
        self.labels[p] = c

        i = 0
        while i < len(neighbors):
            q = neighbors[i]

            if self.labels[q] == -1:
                # Bivsi sum postaje granicna tacka klastera
                self.labels[q] = c
            elif self.labels[q] == 0:
                self.labels[q] = c
                new_neighbors = self.eps_neighbors(q, data)
                if len(new_neighbors) >= self.min_pts:
                    # q je novi jezgreni cvor - prosiri front
                    neighbors = neighbors + new_neighbors

            i += 1

    '''Pokretanje DBSCAN algoritma'''
    def fit(self, data):
        n = len(data)
        self.labels = [0] * n
        c = 0

        for p in range(n):
            if self.labels[p] != 0:
                continue

            neighbors = self.eps_neighbors(p, data)

            if len(neighbors) < self.min_pts:
                self.labels[p] = -1
            else:
                c += 1
                self.expand_cluster(data, p, neighbors, c)

        return self.labels

    '''Broj pronadenih klastera (bez suma)'''
    def num_clusters(self):
        return max(self.labels) if self.labels else 0

    '''Indeksi tacaka oznacenih kao sum'''
    def noise_points(self):
        return [i for i, l in enumerate(self.labels) if l == -1]


if __name__ == '__main__':
    print("=" * 60)
    print("DBSCAN - test na podacima genske ekspresije")
    print("=" * 60)

    print("\nUticaj parametara eps i min_pts:")
    for eps, min_pts in [(0.5, 2), (1.0, 2), (1.5, 2), (2.0, 2), (1.5, 3), (0.5, 3)]:
        db = DBSCAN(eps=eps, min_pts=min_pts)
        labels = db.fit(genes)

        n_clusters = db.num_clusters()
        noise_idx = db.noise_points()

        print(f"\neps={eps}, min_pts={min_pts}  ->  {n_clusters} klastera, {len(noise_idx)} sum")

        for cluster_id in range(1, n_clusters + 1):
            members = [gene_names[i] for i in range(len(genes)) if labels[i] == cluster_id]
            print(f"  Klaster {cluster_id}: {members}")

        if noise_idx:
            noise_genes = [gene_names[i] for i in noise_idx]
            print(f"  Sum: {noise_genes}")
