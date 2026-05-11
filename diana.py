"""
DIANA (Divisive Analysis) — top-down hijerarhijsko klasterovanje.

Komplementarni pristup aglomerativnom hijerarhijskom klasterovanju: umesto
spajanja od dna nagore, DIANA deli klastere od korena nadole.

Algoritam (Kaufman-Rousseeuw):
  1. Krenuti sa jednim klasterom koji sadrzi sve tacke
  2. Pronaci klaster sa najvecim precnikom
  3. U njemu pronaci tacku sa najvecim prosecnim rastojanjem -> osnivac splinter grupe
  4. Iterativno premestati tacke iz ostatka u splinter
  5. Ponavljati 2-4 dok ne ostanu singltoni
"""

from utils import (
    compute_distance_matrix, genes, gene_names, print_clusters_by_labels
)


'''Pomocna klasa za DIANA klaster: visina = precnik u trenutku podele'''
class DCluster:
    def __init__(self, elements, height=0):
        self.elements = elements
        self.height = height
        self.left = None
        self.right = None

    def __str__(self):
        return f'{self.elements}'

    '''Precnik klastera: maksimalno rastojanje izmedu dva njegova clana'''
    @staticmethod
    def diameter(cluster, D):
        max_val = 0.0
        elements = cluster.elements
        for a in range(len(elements)):
            for b in range(a + 1, len(elements)):
                if D[elements[a]][elements[b]] > max_val:
                    max_val = D[elements[a]][elements[b]]
        return max_val


class DIANA:
    def __init__(self):
        self.root = None
        self.history = []

    '''Prosecno rastojanje tacke i do svih tacaka iz skupa S (iskljucujuci samu sebe)'''
    def mean_to_set(self, i, S, D):
        others = [j for j in S if j != i]
        if not others:
            return 0.0
        return sum(D[i][j] for j in others) / len(others)

    '''Podela klastera na splinter grupu i ostatak (Kaufman-Rousseeuw)'''
    def split_cluster(self, cluster, D):
        elements = cluster.elements[:]

        # Osnivac splinter grupe: tacka sa najvecim prosecnim rastojanjem
        max_mean = -1
        founder = elements[0]
        for i in elements:
            p = self.mean_to_set(i, elements, D)
            if p > max_mean:
                max_mean = p
                founder = i

        splinter = [founder]
        rest = [i for i in elements if i != founder]

        # Iterativno premestaj tacke koje su blize splinteru nego ostatku
        changed = True
        while changed and rest:
            changed = False
            best = None
            largest_diff = 0

            for i in rest:
                a = self.mean_to_set(i, rest, D)
                b = self.mean_to_set(i, splinter, D)
                diff = a - b
                if diff > largest_diff:
                    largest_diff = diff
                    best = i

            if best is not None:
                splinter.append(best)
                rest.remove(best)
                changed = True

        return splinter, rest

    '''Pronalazenje klastera sa najvecim precnikom (kandidat za podelu)'''
    def most_diverse_cluster(self, clusters, D):
        candidates = [c for c in clusters if len(c.elements) > 1]
        if not candidates:
            return None, 0.0

        max_d = -1
        chosen = candidates[0]
        for c in candidates:
            p = DCluster.diameter(c, D)
            if p > max_d:
                max_d = p
                chosen = c

        return chosen, max_d

    '''Pokretanje DIANA algoritma — deljenje dok svi klasteri ne postanu singltoni'''
    def fit(self, D):
        n = len(D)
        self.root = DCluster(list(range(n)))
        self.history = []

        clusters = [self.root]

        while True:
            chosen, diameter = self.most_diverse_cluster(clusters, D)
            if chosen is None:
                break

            splinter, rest = self.split_cluster(chosen, D)

            chosen.height = diameter
            chosen.left = DCluster(splinter)
            chosen.right = DCluster(rest)

            self.history.append((splinter, rest, diameter))

            clusters.remove(chosen)
            clusters.append(chosen.left)
            clusters.append(chosen.right)

        return self.root

    '''Izdvajanje k klastera iz dendrograma — secenje na najvisim podelama'''
    def extract_clusters(self, k):
        nodes = [self.root]

        while len(nodes) < k:
            max_h = -1
            max_idx = -1
            for idx, c in enumerate(nodes):
                if c.left is not None and c.height > max_h:
                    max_h = c.height
                    max_idx = idx

            if max_idx < 0:
                break

            cvor = nodes.pop(max_idx)
            nodes.append(cvor.left)
            nodes.append(cvor.right)

        return [c.elements for c in nodes]

    '''Ispis dendrograma (redosled podela, od najvece)'''
    def print_dendrogram(self, names=None):
        for splinter, rest, p in self.history:
            if names:
                splinter_name = [names[x] for x in splinter]
                rest_name = [names[x] for x in rest]
                print(f'  Podela: {splinter_name} | {rest_name}  (precnik: {p:.3f})')
            else:
                print(f'  Podela: {splinter} | {rest}  (precnik: {p:.3f})')

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
    print("DIANA — test na podacima genske ekspresije")
    print("=" * 60)

    D = compute_distance_matrix(genes)

    diana = DIANA()
    diana.fit(D)

    print("\nRedosled podela (od korena nadole):")
    diana.print_dendrogram(gene_names)

    print("\nIzdvojeni klasteri (k=3):")
    clusters_3 = diana.extract_clusters(3)
    for i, cluster in enumerate(clusters_3):
        members = [gene_names[x] for x in cluster]
        print(f"  Klaster {i + 1}: {members}")

    print("\nKlasteri za razlicite k:")
    for k in [2, 3, 4, 5]:
        clusters_k = diana.extract_clusters(k)
        sizes = sorted([len(c) for c in clusters_k], reverse=True)
        print(f"  k={k}: velicine = {sizes}")
