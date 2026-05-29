"""
Pomocne funkcije i deljeni test podaci za algoritme klasterovanja.

Sve implementacije koriste samo standardnu Python biblioteku (math, random).
Komentari su na srpskom bez kvacica, identifikatori na engleskom.
"""

import math


'''Euklidsko rastojanje izmedu dve tacke'''
def euclidean_distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


'''Centar gravitacije skupa tacaka (sredina po svakoj dimenziji)'''
def centroid(points):
    m = len(points[0])
    n = len(points)
    return [sum(p[d] for p in points) / n for d in range(m)]


'''Pearsonov koeficijent korelacije izmedu dva vektora'''
def pearson_correlation(x, y):
    n = len(x)
    if n == 0:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denom_x = math.sqrt(sum((x[i] - mean_x) ** 2 for i in range(n)))
    denom_y = math.sqrt(sum((y[i] - mean_y) ** 2 for i in range(n)))

    if denom_x == 0 or denom_y == 0:
        return 0.0

    return numerator / (denom_x * denom_y)


'''Matrica euklidskih rastojanja izmedu svih parova tacaka'''
def compute_distance_matrix(data):
    n = len(data)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean_distance(data[i], data[j])
            D[i][j] = d
            D[j][i] = d
    return D


'''Matrica Pearsonove korelacije izmedu svih parova vektora'''
def compute_similarity_matrix(data):
    n = len(data)
    R = [[0.0] * n for _ in range(n)]
    for i in range(n):
        R[i][i] = 1.0
        for j in range(i + 1, n):
            r = pearson_correlation(data[i], data[j])
            R[i][j] = r
            R[j][i] = r
    return R


# ===== Test podaci =====
# Simulirani log2 fold change vektori ekspresije za 10 gena kvasca.
# 7 vremenskih tacaka: -6h, -4h, -2h, 0h (diauksicna promena), +2h, +4h, +6h.
# Bazirano na DeRisi eksperimentu (kvasac, prelaz sa fermentacije na respiraciju).

genes = [
    # Up-regulisani geni (respiracija)
    [0.11, 0.43, 0.45, 1.89, 2.00, 3.32, 2.56],
    [0.20, 0.50, 0.60, 1.70, 2.10, 3.00, 2.80],
    [0.05, 0.30, 0.35, 1.50, 1.80, 2.90, 2.30],
    [0.15, 0.40, 0.50, 1.60, 1.90, 3.10, 2.60],
    # Down-regulisani geni (fermentacija)
    [0.09, -0.28, -0.15, -1.18, -1.59, -2.96, -3.08],
    [-0.10, -0.20, -0.30, -1.00, -1.40, -2.50, -2.80],
    [0.05, -0.15, -0.25, -1.30, -1.70, -3.10, -2.90],
    # Nepromenjeni geni (housekeeping)
    [0.15, 0.15, 0.17, 0.09, 0.07, 0.09, 0.07],
    [0.10, 0.08, 0.12, 0.05, 0.03, 0.06, 0.04],
    [-0.05, 0.02, -0.03, 0.08, -0.02, 0.01, 0.03],
]

gene_names = [
    "YLR258W", "GEN_UP1", "GEN_UP2", "GEN_UP3",
    "YPL012W", "GEN_DN1", "GEN_DN2",
    "YPR055W", "GEN_FL1", "GEN_FL2",
]

# Ocekivane grupe (za interpretaciju rezultata):
expected_groups = {
    'Up': ['YLR258W', 'GEN_UP1', 'GEN_UP2', 'GEN_UP3'],
    'Down': ['YPL012W', 'GEN_DN1', 'GEN_DN2'],
    'Flat': ['YPR055W', 'GEN_FL1', 'GEN_FL2'],
}

# Tacke iz Slike 8.6 (desno) - osam tacaka u 2D prostoru iz udzbenika
points_2d = [
    [1, 6], [1, 3], [3, 4], [5, 6], [5, 2], [7, 1], [8, 7], [10, 3]
]


'''Pomocna funkcija za formatirani ispis klastera po imenima'''
def print_clusters_by_labels(labels, names, title=None):
    if title:
        print(title)

    clusters = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(lab, []).append(names[i])

    for cluster_id in sorted(clusters.keys()):
        if cluster_id == -1:
            print(f"  Sum: {clusters[cluster_id]}")
        else:
            print(f"  Klaster {cluster_id}: {clusters[cluster_id]}")


if __name__ == '__main__':
    # Demonstracija pomocnih funkcija
    print("=" * 60)
    print("utils.py - demonstracija pomocnih funkcija")
    print("=" * 60)

    print(f"\nBroj gena: {len(genes)}")
    print(f"Broj vremenskih tacaka: {len(genes[0])}")
    print(f"\nImena gena: {gene_names}")

    print(f"\nEuklidsko rastojanje izmedu gena 0 i 4 (Up vs Down):")
    print(f"  d = {euclidean_distance(genes[0], genes[4]):.4f}")

    print(f"\nPearsonova korelacija izmedu gena 0 i 1 (Up vs Up):")
    print(f"  r = {pearson_correlation(genes[0], genes[1]):.4f}")

    print(f"\nPearsonova korelacija izmedu gena 0 i 4 (Up vs Down):")
    print(f"  r = {pearson_correlation(genes[0], genes[4]):.4f}")

    print(f"\nCentar gravitacije Up gena (geni 0-3):")
    c = centroid(genes[:4])
    print(f"  {[round(x, 3) for x in c]}")

    print(f"\nMatrica rastojanja (3x3 izvod):")
    D = compute_distance_matrix(genes)
    for i in range(3):
        print(f"  {[round(D[i][j], 2) for j in range(3)]}")
