"""
CAST (Cluster Affinity Search Technique).

Zasnovan na konceptu pokvarenih klika - u idealnom slucaju, geni u istom klasteru
bi formirali klike u grafu slicnosti, ali greske u merenjima to narusavaju.
CAST iterativno gradi klastere dodajuci theta-bliske i uklanjajuci theta-udaljene gene.

Broj klastera nije zadat - proizilazi iz strukture podataka i parametra theta.
"""

from utils import pearson_correlation, genes, gene_names


class CAST:
    def __init__(self, theta):
        self.theta = theta
        self.clusters = []
        self.similarity_matrix = None

    '''Izracunavanje matrice slicnosti (Pearsonova korelacija)'''
    def compute_similarity(self, data):
        n = len(data)
        R = [[0.0] * n for _ in range(n)]

        for i in range(n):
            R[i][i] = 1.0
            for j in range(i + 1, n):
                corr = pearson_correlation(data[i], data[j])
                R[i][j] = corr
                R[j][i] = corr

        self.similarity_matrix = R
        return R

    '''Prosecna slicnost gena prema klasteru: R(i, C)'''
    def similarity_with_cluster(self, gene, cluster, R):
        if not cluster:
            return 0.0
        return sum(R[gene][j] for j in cluster) / len(cluster)

    '''Stepen cvora u grafu slicnosti (broj suseda sa R > theta)'''
    def node_degree(self, node, remaining, R):
        return sum(1 for j in remaining if j != node and R[node][j] > self.theta)

    '''Pokretanje CAST algoritma'''
    def fit(self, data=None, similarity_matrix=None):
        if similarity_matrix is not None:
            R = similarity_matrix
            self.similarity_matrix = R
        elif data is not None:
            R = self.compute_similarity(data)
        else:
            raise ValueError("Mora biti dat ili data ili similarity_matrix")

        n = len(R)
        self.clusters = []
        remaining = set(range(n))

        while remaining:
            # Nadji cvor sa maksimalnim stepenom
            max_degree = -1
            starting = None
            for node in remaining:
                deg = self.node_degree(node, remaining, R)
                if deg > max_degree:
                    max_degree = deg
                    starting = node

            cluster = {starting}

            # Iterativno dodaj theta-bliske i ukloni theta-udaljene
            changed = True
            while changed:
                changed = False

                # Dodaj najblizi theta-blizak gen koji nije u C
                best = None
                best_sim = -float('inf')
                for i in remaining:
                    if i in cluster:
                        continue
                    sim = self.similarity_with_cluster(i, cluster, R)
                    if sim > self.theta and sim > best_sim:
                        best_sim = sim
                        best = i

                if best is not None:
                    cluster.add(best)
                    changed = True

                # Ukloni najudaljeniji theta-udaljen gen iz C
                if len(cluster) > 1:
                    worst = None
                    worst_sim = float('inf')
                    for i in list(cluster):
                        others = cluster - {i}
                        sim = self.similarity_with_cluster(i, others, R)
                        if sim <= self.theta and sim < worst_sim:
                            worst_sim = sim
                            worst = i

                    if worst is not None:
                        cluster.remove(worst)
                        changed = True

            self.clusters.append(sorted(cluster))
            remaining -= cluster

        return self.clusters

    '''Labele klastera za svaku tacku (kao lista)'''
    def labels(self, n=None):
        if n is None:
            n = max(max(c) for c in self.clusters) + 1

        lab = [-1] * n
        for cluster_id, cluster in enumerate(self.clusters):
            for gene in cluster:
                lab[gene] = cluster_id
        return lab


if __name__ == '__main__':
    print("=" * 60)
    print("CAST - test na podacima genske ekspresije")
    print("=" * 60)

    cast = CAST(theta=0.7)
    clusters = cast.fit(data=genes)

    print(f"\nCAST (theta=0.7, Pearsonova korelacija)")
    print("-" * 40)
    print(f"Pronadeno {len(clusters)} klastera (automatski!)\n")

    for i, cluster in enumerate(clusters):
        members = [gene_names[x] for x in cluster]
        print(f"  Klaster {i + 1}: {members}")

    print("\nUticaj praga theta:")
    for theta in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        cast_t = CAST(theta=theta)
        clusters_t = cast_t.fit(data=genes)
        sizes = [len(c) for c in clusters_t]
        print(f"  theta={theta:.2f} -> {len(clusters_t)} klastera, velicine: {sizes}")

    print("\nMatrica Pearsonove korelacije (izvod 5x5):")
    R = cast.similarity_matrix
    print(f"{'':>10}", end="")
    for name in gene_names[:5]:
        print(f"{name:>10}", end="")
    print()
    for i in range(5):
        print(f"{gene_names[i]:<10}", end="")
        for j in range(5):
            print(f"{R[i][j]:>10.3f}", end="")
        print()
