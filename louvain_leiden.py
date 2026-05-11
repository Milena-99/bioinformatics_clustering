"""
Louvain i Leiden — community detection algoritmi na grafu slicnosti.

Geni se modeluju kao cvorovi grafa, a tezinske grane se povlace izmedu slicnih
gena (npr. Pearsonova korelacija > prag). Cilj je particija koja maksimizuje
modularnost Q:

    Q = sum_c [ sum_in(c) / 2m  -  (sum_tot(c) / 2m)^2 ]

Louvain — pohlepna lokalna optimizacija: za svaki cvor proba da ga premesti u
zajednicu suseda, prihvata premestanje ako povecava Q.

Leiden — popravlja Louvain dodajuci fazu precistavanja unutar zajednica
(sprecava odvajanje povezanih komponenti).
"""

from utils import compute_similarity_matrix, genes, gene_names


'''Pomocna klasa za reprezentaciju tezinskog grafa slicnosti'''
class SimilarityGraph:
    def __init__(self):
        self.neighbors = {}    # cvor -> {sused: tezina}

    '''Dodavanje cvora bez suseda'''
    def add_node(self, u):
        if u not in self.neighbors:
            self.neighbors[u] = {}

    '''Dodavanje tezinske grane (neusmerene)'''
    def add_edge(self, u, v, w):
        self.add_node(u)
        self.add_node(v)
        if u != v:
            self.neighbors[u][v] = w
            self.neighbors[v][u] = w

    '''Tezinski stepen cvora (zbir tezina incidentnih grana)'''
    def degree(self, u):
        if u not in self.neighbors:
            return 0.0
        return sum(self.neighbors[u].values())

    '''2m: zbir tezina svih grana grafa (svaka brojana dvaput)'''
    def total_weight_2m(self):
        m2 = 0.0
        for u in self.neighbors:
            for w in self.neighbors[u].values():
                m2 += w
        return m2

    '''Konstrukcija grafa iz matrice slicnosti — grane samo gde R > prag'''
    @staticmethod
    def from_similarity_matrix(R, threshold=0.0):
        G = SimilarityGraph()
        n = len(R)
        for u in range(n):
            G.add_node(u)
        for u in range(n):
            for v in range(u + 1, n):
                if R[u][v] > threshold:
                    G.add_edge(u, v, R[u][v])
        return G


class Louvain:
    def __init__(self, similarity_threshold=0.0):
        self.similarity_threshold = similarity_threshold
        self.communities = {}
        self.modularity_value = 0.0

    '''Modularnost particije P (cvor -> zajednica)'''
    def modularity(self, G, P):
        m2 = G.total_weight_2m()
        if m2 == 0:
            return 0.0

        comm_inside = {}    # 2x tezina ivica unutar zajednice
        comm_degree = {}    # ukupni tezinski stepen zajednice

        for u in G.neighbors:
            comm = P[u]
            comm_degree[comm] = comm_degree.get(comm, 0.0) + G.degree(u)
            for v, w in G.neighbors[u].items():
                if P[v] == comm:
                    comm_inside[comm] = comm_inside.get(comm, 0.0) + w

        Q = 0.0
        for comm in set(P.values()):
            inside = comm_inside.get(comm, 0.0)
            total = comm_degree.get(comm, 0.0)
            Q += (inside / m2) - (total / m2) ** 2
        return Q

    '''Jedan prolaz lokalne optimizacije: pomeraj cvor u susednu zajednicu sa najvecim ΔQ'''
    def local_optimization(self, G, P):
        changed_globally = False
        changed = True

        while changed:
            changed = False
            for u in list(G.neighbors.keys()):
                current = P[u]
                neighbor_comms = set(P[v] for v in G.neighbors[u])
                neighbor_comms.add(current)

                Q_current = self.modularity(G, P)
                best = current
                best_delta = 0.0

                for comm in neighbor_comms:
                    if comm == current:
                        continue
                    P[u] = comm
                    delta = self.modularity(G, P) - Q_current
                    if delta > best_delta:
                        best_delta = delta
                        best = comm

                P[u] = best
                if best != current:
                    changed = True
                    changed_globally = True

        return changed_globally

    '''Pre-numerisanje zajednica u uzastopne brojeve (0, 1, 2, ...)'''
    def normalize_labels(self, P):
        mapping = {}
        for u in P:
            if P[u] not in mapping:
                mapping[P[u]] = len(mapping)
        return {u: mapping[P[u]] for u in P}

    '''Pokretanje Louvain algoritma (graf iz Pearsonove korelacije nad podacima)'''
    def fit(self, data):
        R = compute_similarity_matrix(data)
        G = SimilarityGraph.from_similarity_matrix(R, self.similarity_threshold)

        # Inicijalno: svaki cvor je svoja zajednica
        P = {u: u for u in G.neighbors}

        # Pohlepna lokalna optimizacija
        self.local_optimization(G, P)

        self.communities = self.normalize_labels(P)
        self.modularity_value = self.modularity(G, self.communities)
        return self.communities

    '''Labele zajednica kao lista po indeksu tacke'''
    def labels(self, n):
        return [self.communities.get(i, -1) for i in range(n)]


class Leiden(Louvain):
    '''Faza precistavanja: unutar svake zajednice ponovo pokreni lokalnu optimizaciju'''
    def refine(self, G, P):
        # Grupisi cvorove po trenutnoj zajednici
        comm_members = {}
        for u, comm in P.items():
            comm_members.setdefault(comm, []).append(u)

        new_P = {}
        next_id = 0

        for comm, members in comm_members.items():
            # Indukovani podgraf samo nad clanovima ove zajednice
            subgraph = SimilarityGraph()
            for u in members:
                subgraph.add_node(u)
            for u in members:
                for v, w in G.neighbors[u].items():
                    if v in members and v > u:
                        subgraph.add_edge(u, v, w)

            # Inicijalno svaki cvor podgrafa je svoja podzajednica
            sub_P = {u: u for u in members}

            # Pokusaj lokalnu optimizaciju unutar zajednice
            self.local_optimization(subgraph, sub_P)

            # Mapiraj novo-detektovane podzajednice u globalne ID-jeve
            local_ids = {}
            for u in members:
                sub_comm = sub_P[u]
                if sub_comm not in local_ids:
                    local_ids[sub_comm] = next_id
                    next_id += 1
                new_P[u] = local_ids[sub_comm]

        return new_P

    '''Pokretanje Leiden algoritma: Louvain + faza precistavanja'''
    def fit(self, data):
        R = compute_similarity_matrix(data)
        G = SimilarityGraph.from_similarity_matrix(R, self.similarity_threshold)

        P = {u: u for u in G.neighbors}
        self.local_optimization(G, P)

        # Faza precistavanja
        P = self.refine(G, P)

        # Jos jedan prolaz lokalne optimizacije nad precistenom particijom
        self.local_optimization(G, P)

        self.communities = self.normalize_labels(P)
        self.modularity_value = self.modularity(G, self.communities)
        return self.communities


if __name__ == '__main__':
    print("=" * 60)
    print("Louvain i Leiden — test na podacima genske ekspresije")
    print("=" * 60)

    for threshold in [0.3, 0.5, 0.7, 0.9]:
        print(f"\nPrag slicnosti (Pearson): {threshold}")
        print("-" * 40)

        # Louvain
        lv = Louvain(similarity_threshold=threshold)
        lv.fit(genes)
        print(f"Louvain → Q = {lv.modularity_value:.4f}")

        comm_groups = {}
        for i, z in enumerate(lv.labels(len(genes))):
            comm_groups.setdefault(z, []).append(gene_names[i])
        for comm_id, members in sorted(comm_groups.items()):
            print(f"  Zajednica {comm_id}: {members}")

        # Leiden
        ld = Leiden(similarity_threshold=threshold)
        ld.fit(genes)
        print(f"\nLeiden  → Q = {ld.modularity_value:.4f}")

        comm_groups = {}
        for i, z in enumerate(ld.labels(len(genes))):
            comm_groups.setdefault(z, []).append(gene_names[i])
        for comm_id, members in sorted(comm_groups.items()):
            print(f"  Zajednica {comm_id}: {members}")
