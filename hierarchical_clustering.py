"""
Hijerarhijsko klasterovanje (aglomerativno, bottom-up).

Gradi stablo (dendrogram) spajanjem najblizih klastera. Podrzava dve metode:
  - D_min:  minimalno rastojanje izmedu klastera (single linkage)
  - D_avg:  prosecno rastojanje izmedu klastera (UPGMA, average linkage)

Bioinformaticki kontekst: rekonstrukcija evolucione istorije gena,
identifikacija genskih familija, analiza funkcijskih grupa.
"""

from utils import (
    euclidean_distance, compute_distance_matrix,
    genes, gene_names, print_clusters_by_labels
)


'''Pomocna klasa za reprezentaciju klastera u hijerarhijskom klasterovanju'''
class HCluster:
    def __init__(self, elements, height=0):
        self.elements = elements
        self.height = height
        self.left = None
        self.right = None

    def __str__(self):
        return f'{self.elements}'

    '''D_min: minimalno rastojanje izmedu dva klastera (single linkage)'''
    @staticmethod
    def d_min(c1, c2, D):
        min_r = float('inf')
        for i in c1.elements:
            for j in c2.elements:
                if D[i][j] < min_r:
                    min_r = D[i][j]
        return min_r

    '''D_avg: prosecno rastojanje izmedu dva klastera (UPGMA)'''
    @staticmethod
    def d_avg(c1, c2, D):
        total = 0
        for i in c1.elements:
            for j in c2.elements:
                total += D[i][j]
        return total / (len(c1.elements) * len(c2.elements))


class HierarchicalClustering:
    def __init__(self, method='avg'):
        self.method = method    # 'min' ili 'avg'
        self.root = None
        self.history = []      # redosled spajanja: (clanovi_1, clanovi_2, rastojanje)

    '''Pronalazak dva najbliza klastera u trenutnoj listi'''
    def two_closest(self, clusters, D):
        min_r = float('inf')
        min_i, min_j = 0, 1

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if self.method == 'min':
                    r = HCluster.d_min(clusters[i], clusters[j], D)
                else:
                    r = HCluster.d_avg(clusters[i], clusters[j], D)

                if r < min_r:
                    min_r = r
                    min_i, min_j = i, j

        return min_i, min_j, min_r

    '''Pokretanje hijerarhijskog klasterovanja na matrici rastojanja'''
    def fit(self, D):
        n = len(D)
        clusters = [HCluster([i], 0) for i in range(n)]
        self.history = []

        while len(clusters) > 1:
            i, j, distance = self.two_closest(clusters, D)

            ci = clusters[i]
            cj = clusters[j]
            new = HCluster(ci.elements + cj.elements, distance / 2)
            new.left = ci
            new.right = cj

            self.history.append((ci.elements, cj.elements, distance))

            # Ukloni stara dva, dodaj novi spojeni
            clusters = [clusters[x] for x in range(len(clusters)) if x != i and x != j]
            clusters.append(new)

        self.root = clusters[0]
        return self.root

    '''Izdvajanje k klastera iz dendrograma — secenje stabla na najvisim cvorovima'''
    def extract_clusters(self, k):
        nodes = [self.root]

        while len(nodes) < k:
            # Pronadi cvor sa najvecom visinom (za sledece deljenje)
            max_h = -1
            max_idx = 0
            for idx, c in enumerate(nodes):
                if c.left is not None and c.height > max_h:
                    max_h = c.height
                    max_idx = idx

            cvor = nodes[max_idx]
            if cvor.left is None:
                break

            nodes.pop(max_idx)
            nodes.append(cvor.left)
            nodes.append(cvor.right)

        return [c.elements for c in nodes]

    '''Ispis dendrograma — redosled spajanja'''
    def print_dendrogram(self, names=None):
        for ci, cj, dist in self.history:
            if names:
                name_i = [names[x] for x in ci]
                name_j = [names[x] for x in cj]
                print(f'  Spoj {name_i} + {name_j}  (rastojanje: {dist:.3f})')
            else:
                print(f'  Spoj {ci} + {cj}  (rastojanje: {dist:.3f})')

    '''Pretvaranje hijerarhijske strukture u flat labele za k klastera'''
    def labels(self, k, n_points):
        clusters_k = self.extract_clusters(k)
        labels = [0] * n_points
        for cluster_id, cluster in enumerate(clusters_k):
            for idx in cluster:
                labels[idx] = cluster_id
        return labels


if __name__ == '__main__':
    print("=" * 60)
    print("Hijerarhijsko klasterovanje — test na podacima genske ekspresije")
    print("=" * 60)

    D = compute_distance_matrix(genes)

    for method, title in [('avg', 'D_avg / UPGMA'), ('min', 'D_min / single linkage')]:
        print(f"\nMetoda: {title}")
        print("-" * 40)

        hc = HierarchicalClustering(method=method)
        hc.fit(D)

        print("Redosled spajanja:")
        hc.print_dendrogram(gene_names)

        print("\nIzdvojeni klasteri (k=3):")
        clusters_3 = hc.extract_clusters(3)
        for i, cluster in enumerate(clusters_3):
            members = [gene_names[x] for x in cluster]
            print(f"  Klaster {i + 1}: {members}")
