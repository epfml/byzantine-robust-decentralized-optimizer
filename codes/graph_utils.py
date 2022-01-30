import numpy as np


class Node(object):
    def __init__(self, index):
        self.index = index
        self.edges = set()

    def add_edge(self, edge):
        assert edge not in self.edges
        self.edges.add(edge)

    @property
    def degree(self):
        return len(self.edges)


class Edge(object):
    def __init__(self, n1, n2):
        assert n1 != n2
        self.nodes = set([n1, n2])

    def theother(self, node: Node):
        assert node in self.nodes
        return (self.nodes - set([node])).pop()


def metropolis_weight(graph):
    mixing = np.zeros((graph.n, graph.n))
    for e in graph.edges:
        n1, n2 = e.nodes
        w = 1.0 / (1 + max(n1.degree, n2.degree))
        mixing[n1.index, n2.index] = w
        mixing[n2.index, n1.index] = w

    for n in graph.nodes:
        mixing[n.index, n.index] = 1 - mixing[n.index, :].sum()

    return mixing


def spectral_gap(mixing: np.array):
    eigenvalues = sorted(np.abs(np.linalg.eigvals(mixing)))
    return 1 - eigenvalues[-2]


class Graph(object):
    def __init__(self, n, edges, metropolis=True):
        assert n > 0 and isinstance(n, int)
        assert isinstance(edges, list)
        self.n = n
        self.nodes = [Node(i) for i in range(n)]

        self.edges = []
        for e in edges:
            node1 = self.nodes[e[0]]
            node2 = self.nodes[e[1]]
            e = Edge(node1, node2)

            node1.add_edge(e)
            node2.add_edge(e)
            self.edges.append(e)

        if metropolis:
            self.metropolis_weight = metropolis_weight(self)
            if n > 1:
                self.spectral_gap = spectral_gap(self.metropolis_weight)
            else:
                self.spectral_gap = 1


class Ring(Graph):
    def __init__(self, n):
        edges = [(i, i + 1) for i in range(n - 1)] + [(n - 1, 0)]
        super().__init__(n, edges)

    def __str__(self):
        return f"Ring(n={self.n})"


class Chain(Graph):
    def __init__(self, n):
        edges = [(i, i + 1) for i in range(n - 1)]
        super().__init__(n, edges)

    def __str__(self):
        return f"Chain(n={self.n})"


class Complete(Graph):
    def __init__(self, n):
        edges = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                edges.append((i, j))
        super().__init__(n, edges)

    def __str__(self):
        return f"Complete(n={self.n})"


class Star(Graph):
    def __init__(self, n):
        edges = [(0, i) for i in range(1, n)]
        super().__init__(n, edges)

    def __str__(self):
        return f"Star(n={self.n})"


class BinaryTree(Graph):
    def __init__(self, n):
        edges = []
        for i in range(n):
            if 2 * i + 1 < n:
                edges.append((i, 2 * i + 1))

            if 2 * i + 2 < n:
                edges.append((i, 2 * i + 2))

        super().__init__(n, edges)

    def __str__(self):
        return f"BinaryTree(n={self.n})"


class TorusGraph(Graph):
    """
    https://mathworld.wolfram.com/CycleGraph.html
    https://mathworld.wolfram.com/TorusGridGraph.html

    Torus graph is catesian product of two rings
    """

    def __init__(self, n, c1=None, c2=None):
        if c1 is not None and c2 is not None:
            assert n == c1 * c2

        edges = []
        for i in range(c1):
            # Add a chain of size c2-1
            edges += [(i * c2 + j, i * c2 + j + 1) for j in range(c2 - 1)]
            # Add one more edge to make it a ring of c2
            edges += [(i * c2 + c2 - 1, i * c2)]

        # Adding edges across different rings
        for j in range(c2):
            edges += [(i * c2 + j, (i + 1) * c2 + j) for i in range(c1 - 1)]
            edges += [((c1 - 1) * c2 + j, j)]

        super().__init__(n, edges)


class TwoCliques(Graph):
    """
    Two cliques (fully connected within clique) connected by m nodes (m \ge 0)
    There are 2n+m nodes in total
    """

    def __init__(self, n, m):
        # n is the size of each clique
        # m is the number of nodes between two cliques
        # clique 1: 0,1, ..., n-1
        # clique 2: n,n+1, ..., 2n-1
        # Connection nodes: 2n, ..., 2n+m-1
        edges = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Add first clique
                edges.append((i, j))
                # Add second clique
                edges.append((i + n, j + n))

        for i in range(2 * n - 1, 2 * n + m - 1):
            edges.append((i, i + 1))
        edges.append((0, 2 * n + m - 1))
        super().__init__(2 * n + m, edges)

    def __str__(self):
        return f"TwoCliques(n={self.n},m={self.m})"


class TwoCliquesWithByzantine(Graph):
    """
    Topology:
    - There are two cliques of same size (`m`)
        - worker_id in clique 1: 0, 1, ..., m-1
        - worker_id in clique 2: m, m+1, ..., 2m-1
    - Two cliques are connected by 1 node (ID 2m) => neighbor 0, 2m-1
    - There are `b` Byzantine nodes connected to node 2m: namely, 2m+1, ..., 2m+b

    Mixing matrix:
        - When there is no Byzantine worker (b=0), we use metropolis weighting 
            - node 2m: gives weight 1/(m+1) to node 0 and node 2m separately and (m-1)/(m+1) to self (here we assume m-1 \ge 2)
        - Assuming that `b` + 2 <= m, then
            - all other weights not related to node 2m remain the same (as if there is no byz)
            - node 2m: gives weight 1/(m+1) to node 0 and node 2m separately
            - node 2m: gives self weight `1 - b / (b + 3) - 2 / (m+1)=(3m-2b-3)/(m+1)(b+3)`
            - node 2m: give each byzantine worker 1 / (b + 3) weight.

    Note that delta\in[0,1] is a coefficient in front of Byzantine weights of central
    node. When delta=0, there is no weight on Byzantine workers and this is equivalent
    to no Byzantine case. However, the changes in delta\in[0,1] does not influence
    the spectral gap of the \wildetilde{W} but only changes \delta_\max = \delta * b / (b + 3)
    """

    def __init__(self, m, b, delta=1):
        assert b + 2 <= m, f"b={b} > m={m} - 2"
        # n is the size of each clique
        # m is the number of nodes between two cliques
        # clique 1: 0,1, ..., m-1
        # clique 2: m,m+1, ..., 2m-1
        # Connection nodes: 2m
        edges = []
        for i in range(m - 1):
            for j in range(i + 1, m):
                # Add first clique
                edges.append((i, j))
                # Add second clique
                edges.append((i + m, j + m))

        edges.append((0, 2 * m))
        edges.append((2 * m - 1, 2 * m))

        for i in range(1, b + 1):
            edges.append((2 * m, 2 * m + i))

        self.m = m
        self.b = b

        # super().__init__(2 * m + 1 + b, edges, metropolis=True)
        super().__init__(2 * m + 1 + b, edges, metropolis=False)

        self.metropolis_weight = metropolis_weight(self)
        c = 2 * m
        self.metropolis_weight[c, c] = 1 - delta * b / (b + 3) - 2 / (m+1)
        for i in range(2*m+1, 2*m+b+1):
            self.metropolis_weight[i, c] = delta / (b + 3)
            self.metropolis_weight[c, i] = delta / (b + 3)
            self.metropolis_weight[i, i] = 1 - delta / (b + 3)

        # Update the corresponding weights at the
        self.spectral_gap = spectral_gap(self.metropolis_weight)

    def __str__(self):
        return f"TwoCliquesWithByzantine(m={self.m},b={self.b})"

    def clique1(self):
        return list(range(self.m))

    def clique2(self):
        return list(range(self.m, self.m * 2))


class Dumbbell(Graph):
    """
    Node 0, ...., 2 * q - 1: good
    Node 2 * q, ..., 2 * q + 2 * qb - 1: Byzantine
    """

    def __init__(self, q, qb, r=0):
        # q: clique_size
        # b: qb
        assert q > 0
        n = 2 * (q + qb)

        mixing = np.zeros((n, n))
        w = 1 / (q + qb + 1 + r)

        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                edges.append((i, j))
                edges.append((i + q, j + q))
                mixing[i, j] = w
                mixing[j, i] = w
                mixing[i + q, j + q] = w
                mixing[j + q, i + q] = w

        # Bridge
        edges.append((0, 2 * q - 1))
        mixing[0, 2 * q - 1] = w
        mixing[2 * q - 1, 0] = w

        # Byzantine edges
        for i in range(qb):
            edges.append((0, 2 * q + i))
            edges.append((2 * q - 1, 2 * q + qb + i))
            mixing[0, 2 * q + i] = w
            mixing[2 * q + i, 0] = w
            mixing[2 * q - 1, 2 * q + qb + i] = w
            mixing[2 * q + qb + i, 2 * q - 1] = w

        # random edges
        endpoint1 = np.random.randint(0, q, size=r)
        endpoint2 = np.random.randint(q, 2 * q, size=r)
        for i, j in zip(endpoint1, endpoint2):
            edges.append((i, j))
            mixing[i, j] = w
            mixing[j, i] = w

        # Add diagonals
        mixing += np.diag(1 - mixing.sum(axis=1))
        super().__init__(n=n, edges=edges, metropolis=False)

        self.metropolis_weight = mixing
        self.w_tilde = self._get_w_tilde(m=mixing, q=q)
        self.spectral_gap = spectral_gap(self.w_tilde)

        self.b = 2 * qb

    def _get_w_tilde(self, m, q):
        w_tilde = m[:2*q, :2*q].copy()
        w_tilde += np.diag(m[:2 * q, 2 * q:].sum(axis=1))
        return w_tilde

    def clique1(self):
        return list(range((self.n - self.b) // 2))

    def clique2(self):
        return list(range((self.n - self.b) // 2, self.n - self.b))


def get_graph(args):
    if args.graph == "ring":
        return Ring(n=args.n)

    if args.graph == "complete":
        return Complete(n=args.n)

    if args.graph == "chain":
        return Chain(n=args.n)

    if args.graph == "star":
        return Star(n=args.n)

    if args.graph == "binarytree":
        return BinaryTree(n=args.n)

    if args.graph.startswith("twocliques"):
        # Pattern: twocliques2,1 for n=2 m=1
        n, m = args.graph[len("twocliques"):].split(",")
        n, m = int(n), int(m)
        assert args.n == 2 * n + m
        return TwoCliques(n, m)

    if args.graph.startswith("torus"):
        # the graph pattern is torusC{}C{}
        c1, c2 = args.graph[6:].split("C")
        return TorusGraph(n=args.n, c1=int(c1), c2=int(c2))

    if args.graph.startswith("dumbbell"):
        q, b, r = args.graph[len("dumbbell"):].split(",")
        q, b, r = int(q), int(b), int(r)
        assert args.n == 2 * (q + b)
        assert args.f == 2 * b
        return Dumbbell(q, b, r)

    raise NotImplementedError(f"{args.graph}")
