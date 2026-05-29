"""
Uporedni testovi svih implementiranih algoritama klasterovanja.

Svi algoritmi se pokrecu na istom skupu podataka (geni kvasca, diauksicna promena)
i njihove particije se uporedjuju u jedinstvenoj tabeli.

Ocekivane grupe:
  Up:   [YLR258W, GEN_UP1, GEN_UP2, GEN_UP3]   (respiracija)
  Down: [YPL012W, GEN_DN1, GEN_DN2]            (fermentacija)
  Flat: [YPR055W, GEN_FL1, GEN_FL2]            (housekeeping)
"""

import random

from utils import (
    genes, gene_names, points_2d, expected_groups,
    compute_distance_matrix, compute_similarity_matrix,
)
from farthest_first_traversal import FarthestFirstTraversal
from k_means import KMeans
from soft_k_means import SoftKMeans
from hierarchical_clustering import HierarchicalClustering
from cast import CAST
from dbscan import DBSCAN
from diana import DIANA
from louvain_leiden import Louvain, Leiden
from brute_force_k_means import BruteForceKMeans


'''Formatiranje klastera u listu sortiranu po prvom clanu (radi uporedivosti)'''
def format_clusters(labels, names, k=None):
    if k is None:
        k = max(labels) + 1 if labels else 0
    result = []
    for cluster_id in range(k):
        members = [names[i] for i in range(len(names)) if labels[i] == cluster_id]
        result.append(members)
    result.sort(key=lambda x: x[0] if x else "")
    return result


'''Skracivanje imena gena za prikaz u tabeli'''
def shorten(names_list, n=6):
    return ", ".join(name[:n] for name in names_list)


'''Pokretanje svih algoritama na istim podacima i formiranje tabele rezultata'''
def run_all_on_genes():
    print("=" * 80)
    print("UPOREDNI TEST: 10 gena kvasca x 7 vremenskih tacaka")
    print("=" * 80)

    D = compute_distance_matrix(genes)
    n = len(genes)
    k = 3

    results = []

    # 1. FFT
    fft = FarthestFirstTraversal(k=k)
    fft.fit(genes)
    fft_labels = fft.assign_clusters(genes)
    results.append(("FarthestFirstTraversal", format_clusters(fft_labels, gene_names, k)))

    # 2. K-means (Lloyd)
    random.seed(42)
    km = KMeans(k=k)
    km.fit(genes)
    results.append((f"Lloyd K-means (D={km.distortion(genes):.4f})",
                    format_clusters(km.labels, gene_names, k)))

    # 3. Meki K-means
    random.seed(42)
    skm = SoftKMeans(k=k, beta=2.0)
    skm.fit(genes)
    results.append(("Meki K-means (beta=2)",
                    format_clusters(skm.hard_labels(), gene_names, k)))

    # 4. Hijerarhijsko (UPGMA)
    hc = HierarchicalClustering(method='avg')
    hc.fit(D)
    hc_labels = hc.labels(k, n)
    results.append(("Hijerarhijsko (UPGMA)", format_clusters(hc_labels, gene_names, k)))

    # 5. Hijerarhijsko (D_min)
    hc_min = HierarchicalClustering(method='min')
    hc_min.fit(D)
    hc_min_labels = hc_min.labels(k, n)
    results.append(("Hijerarhijsko (D_min)", format_clusters(hc_min_labels, gene_names, k)))

    # 6. CAST
    cast = CAST(theta=0.7)
    cast.fit(data=genes)
    cast_labels = cast.labels(n)
    cast_k = max(cast_labels) + 1
    results.append((f"CAST (theta=0.7, {cast_k} klastera)",
                    format_clusters(cast_labels, gene_names, cast_k)))

    # 7. DBSCAN
    db = DBSCAN(eps=1.5, min_pts=2)
    db_labels_raw = db.fit(genes)
    # DBSCAN koristi 1-indeksirane klastere i -1 za sum; konvertuj u 0-indeksirane
    db_labels = [l - 1 if l > 0 else -1 for l in db_labels_raw]
    db_k = max(db_labels) + 1
    results.append((f"DBSCAN (eps=1.5, mp=2)",
                    format_clusters(db_labels, gene_names, db_k)))

    # 8. DIANA
    diana = DIANA()
    diana.fit(D)
    diana_labels = diana.labels(k, n)
    results.append(("DIANA (divisive)", format_clusters(diana_labels, gene_names, k)))

    # 9. Louvain
    lv = Louvain(similarity_threshold=0.5)
    lv.fit(genes)
    lv_labels = lv.labels(n)
    lv_k = max(lv_labels) + 1
    results.append((f"Louvain (Q={lv.modularity_value:.3f})",
                    format_clusters(lv_labels, gene_names, lv_k)))

    # 10. Leiden
    ld = Leiden(similarity_threshold=0.5)
    ld.fit(genes)
    ld_labels = ld.labels(n)
    ld_k = max(ld_labels) + 1
    results.append((f"Leiden (Q={ld.modularity_value:.3f})",
                    format_clusters(ld_labels, gene_names, ld_k)))

    # 11. Brute-force K-means
    bfkm = BruteForceKMeans(k=k)
    bfkm.fit(genes)
    results.append((f"Brute-force K-means (D={bfkm.best_distortion:.4f})",
                    format_clusters(bfkm.labels, gene_names, k)))

    # Ispis tabele
    print()
    header = f"{'Algoritam':<32} {'Klaster 1':<22} {'Klaster 2':<22} {'Klaster 3':<22}"
    print(header)
    print("-" * len(header))

    for name, clusters in results:
        cols = []
        for cluster in clusters[:3]:
            cols.append(shorten(cluster))
        while len(cols) < 3:
            cols.append("-")
        # Ako ima vise od 3 klastera, dodaj indikator
        extra = ""
        if len(clusters) > 3:
            extra = f"  (+{len(clusters) - 3})"
        print(f"{name:<32} {cols[0]:<22} {cols[1]:<22} {cols[2]:<22}{extra}")

    print()
    print("Ocekivano:")
    print("  Up:   YLR258, GEN_UP, GEN_UP, GEN_UP")
    print("  Down: YPL012, GEN_DN, GEN_DN")
    print("  Flat: YPR055, GEN_FL, GEN_FL")


'''Test na 2D tackama iz Slike 8.6 (udzbenik)'''
def run_on_2d_points():
    print("\n" + "=" * 80)
    print("UPOREDNI TEST: 8 tacaka u 2D prostoru (Slika 8.6, udzbenik)")
    print("=" * 80)

    print("\nTacke:")
    for i, p in enumerate(points_2d):
        print(f"  T{i}: ({p[0]}, {p[1]})")

    D = compute_distance_matrix(points_2d)
    n = len(points_2d)
    k = 3

    # FFT
    fft = FarthestFirstTraversal(k=k)
    fft.fit(points_2d)
    fft_labels = fft.assign_clusters(points_2d)
    print(f"\nFFT (k=3):")
    for c in range(k):
        members = [f"T{i}" for i in range(n) if fft_labels[i] == c]
        print(f"  Klaster {c + 1}: {members}")

    # K-means
    random.seed(0)
    km = KMeans(k=k)
    km.fit(points_2d)
    print(f"\nLloyd K-means (k=3, D={km.distortion(points_2d):.4f}):")
    for c in range(k):
        members = [f"T{i}" for i in range(n) if km.labels[i] == c]
        print(f"  Klaster {c + 1}: {members}")

    # Hijerarhijsko UPGMA
    hc = HierarchicalClustering(method='avg')
    hc.fit(D)
    print(f"\nHijerarhijsko UPGMA - redosled spajanja:")
    hc.print_dendrogram()
    clusters_3 = hc.extract_clusters(3)
    print(f"\n  3 klastera: {clusters_3}")


'''Sazetak: koji algoritmi su pronasli ocekivanu particiju (Up/Down/Flat)?'''
def correctness_summary():
    print("\n" + "=" * 80)
    print("SAZETAK ISPRAVNOSTI")
    print("=" * 80)

    expected_sets = [set(g) for g in expected_groups.values()]

    def is_correct(clusters):
        clusters_as_sets = [set(c) for c in clusters if c]
        if len(clusters_as_sets) != 3:
            return False
        return all(s in clusters_as_sets for s in expected_sets)

    n = len(genes)
    D = compute_distance_matrix(genes)
    k = 3

    print()
    tests = []

    # FFT
    fft = FarthestFirstTraversal(k=k)
    fft.fit(genes)
    fft_labels = fft.assign_clusters(genes)
    tests.append(("FarthestFirstTraversal", format_clusters(fft_labels, gene_names, k)))

    # K-means (5 seed-ova)
    km_correct = 0
    for seed in [0, 1, 7, 42, 100]:
        random.seed(seed)
        km = KMeans(k=k)
        km.fit(genes)
        if is_correct(format_clusters(km.labels, gene_names, k)):
            km_correct += 1

    # Hijerarhijsko UPGMA
    hc = HierarchicalClustering(method='avg')
    hc.fit(D)
    tests.append(("Hijerarhijsko (UPGMA)", format_clusters(hc.labels(k, n), gene_names, k)))

    # Hijerarhijsko D_min
    hc_min = HierarchicalClustering(method='min')
    hc_min.fit(D)
    tests.append(("Hijerarhijsko (D_min)", format_clusters(hc_min.labels(k, n), gene_names, k)))

    # DBSCAN
    db = DBSCAN(eps=1.5, min_pts=2)
    db_raw = db.fit(genes)
    db_labels = [l - 1 if l > 0 else -1 for l in db_raw]
    db_k = max(db_labels) + 1
    tests.append(("DBSCAN (eps=1.5, mp=2)", format_clusters(db_labels, gene_names, db_k)))

    # DIANA
    diana = DIANA()
    diana.fit(D)
    tests.append(("DIANA", format_clusters(diana.labels(k, n), gene_names, k)))

    # Louvain
    lv = Louvain(similarity_threshold=0.5)
    lv.fit(genes)
    lv_k = max(lv.labels(n)) + 1
    tests.append(("Louvain (theta=0.5)", format_clusters(lv.labels(n), gene_names, lv_k)))

    # Brute-force
    bfkm = BruteForceKMeans(k=k)
    bfkm.fit(genes)
    tests.append(("Brute-force K-means", format_clusters(bfkm.labels, gene_names, k)))

    for name, clusters in tests:
        ok = "[x]" if is_correct(clusters) else "[ ]"
        print(f"  {ok}  {name}")

    print(f"  Lloyd K-means: {km_correct}/5 seed-ova nasli tacnu particiju")


if __name__ == '__main__':
    run_all_on_genes()
    run_on_2d_points()
    correctness_summary()
