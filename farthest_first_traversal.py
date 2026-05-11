"""
FarthestFirstTraversal — heuristika za k-centar problem klasterovanja.

Bira k centara iz skupa tacaka tako da minimizuje maksimalno rastojanje
bilo koje tacke do njenog najblizeg centra. Garantuje 2-aproksimaciju.
"""

from utils import euclidean_distance, genes, gene_names, print_clusters_by_labels


class FarthestFirstTraversal:
    def __init__(self, k):
        self.k = k
        self.centers = []

    '''Rastojanje tacke do najblizeg vec izabranog centra'''
    def distance_to_centers(self, point):
        return min(euclidean_distance(point, c) for c in self.centers)

    '''Pokretanje algoritma — bira k tacaka kao centre'''
    def fit(self, data):
        # Izaberi prvu tacku proizvoljno
        self.centers = [data[0][:]]

        while len(self.centers) < self.k:
            # Pronadi tacku najudaljeniju od svih vec izabranih centara
            max_dist = -1
            farthest = None

            for point in data:
                r = self.distance_to_centers(point)
                if r > max_dist:
                    max_dist = r
                    farthest = point

            self.centers.append(farthest[:])

        return self.centers

    '''Dodela tacaka klasterima po najblizem centru'''
    def assign_clusters(self, data):
        labels = []
        for point in data:
            min_r = float('inf')
            closest = 0
            for i, c in enumerate(self.centers):
                r = euclidean_distance(point, c)
                if r < min_r:
                    min_r = r
                    closest = i
            labels.append(closest)
        return labels


if __name__ == '__main__':
    print("=" * 60)
    print("FarthestFirstTraversal — test na podacima genske ekspresije")
    print("=" * 60)

    fft = FarthestFirstTraversal(k=3)
    fft.fit(genes)
    labels = fft.assign_clusters(genes)

    print("\nFarthestFirstTraversal (k=3)")
    print("-" * 40)
    print_clusters_by_labels(labels, gene_names)

    print("\nIzabrani centri (tacke iz podataka):")
    for i, c in enumerate(fft.centers):
        print(f"  Centar {i + 1}: [{', '.join(f'{x:+.2f}' for x in c)}]")

    print("\nOcekivane grupe:")
    print("  Up:   ['YLR258W', 'GEN_UP1', 'GEN_UP2', 'GEN_UP3']")
    print("  Down: ['YPL012W', 'GEN_DN1', 'GEN_DN2']")
    print("  Flat: ['YPR055W', 'GEN_FL1', 'GEN_FL2']")
