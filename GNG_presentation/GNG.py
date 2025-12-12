import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

class GrowingNeuralGas:
    def __init__(self,
                 epsilon_b=0.05,      # reduced from 0.2 → more stable
                 epsilon_n=0.0006,
                 lambda_insert=100,   # insert every 100 steps
                 max_age=88,          # standard value in literature
                 alpha=0.5,          # error reduction for q and f
                 beta=0.0005):        # error decay per step (very small)
        self.epsilon_b = epsilon_b
        self.epsilon_n = epsilon_n
        self.lambda_insert = lambda_insert
        self.max_age = max_age
        self.alpha = alpha
        self.beta = beta

        # Initialize with 2 distinct neurons inside the data range
        np.random.seed(123)  # better seed → visible separation
        self.units = np.random.uniform(-0.5, 1.5, size=(2, 2))
        self.units[0] = [0.0, 0.5]   # force visible separation
        self.units[1] = [1.0, 0.5]

        self.errors = np.zeros(2)
        self.edges = set()
        self.edge_ages = {}
        self.step = 0

    def find_bmu(self, x):
        dists = np.linalg.norm(self.units - x, axis=1)
        s1 = np.argmin(dists)
        dists[s1] = np.inf
        s2 = np.argmin(dists)
        return s1, s2

    def get_neighbors(self, idx):
        return [j for (i, j) in self.edges if i == idx] + \
               [i for (i, j) in self.edges if j == idx]

    def train_one(self, x):
        self.step += 1
        s1, s2 = self.find_bmu(x)

        # --- Adaptation ---
        self.units[s1] += self.epsilon_b * (x - self.units[s1])
        for n in self.get_neighbors(s1):
            self.units[n] += self.epsilon_n * (x - self.units[n])

        # --- Local error ---
        self.errors[s1] += np.sum((x - self.units[s1])**2)

        # --- Topology: connect s1 and s2, reset/increase ages ---
        edge = tuple(sorted((s1, s2)))
        if edge not in self.edges:
            self.edges.add(edge)
            self.edge_ages[edge] = 0
        else:
            self.edge_ages[edge] = 0

        # Increase age of all edges from winner
        for e in list(self.edges):
            if s1 in e:
                self.edge_ages[e] += 1

        # Remove old edges
        old_edges = [e for e in self.edges if self.edge_ages[e] > self.max_age]
        for e in old_edges:
            self.edges.discard(e)
            del self.edge_ages[e]

        # Remove isolated nodes (optional)
        active_nodes = {i for e in self.edges for i in e}
        isolated = [i for i in range(len(self.units)) if i not in active_nodes]
        if isolated and len(self.units) > 2:
            # only remove if we have more than 2 nodes
            mask = np.ones(len(self.units), dtype=bool)
            mask[isolated] = False
            self.units = self.units[mask]
            self.errors = self.errors[mask]
            # rebuild edges with new indices (simple but safe for small N)
            new_edges = set()
            new_ages = {}
            mapping = {old: new for new, old in enumerate(np.where(mask)[0])}
            for (a, b) in self.edges:
                if a in mapping and b in mapping:
                    new_edge = tuple(sorted((mapping[a], mapping[b])))
                    new_edges.add(new_edge)
                    new_ages[new_edge] = self.edge_ages[(a, b)]
            self.edges = new_edges
            self.edge_ages = new_ages

        # --- Insert new unit every lambda steps ---
        if self.step % self.lambda_insert == 0:
            self._insert_unit()

        # --- Global error decay (only after possible insertion) ---
        self.errors -= self.beta * self.errors  # subtractive form = multiplicative with d=1-beta

    def _insert_unit(self):
        if len(self.units) < 2:
            return
        q = np.argmax(self.errors)
        neighbors = self.get_neighbors(q)
        if not neighbors:
            return
        f = neighbors[np.argmax([self.errors[n] for n in neighbors])]

        r = len(self.units)
        new_unit = 0.5 * (self.units[q] + self.units[f])
        self.units = np.vstack([self.units, new_unit])

        # Proper error for new node: average of q and f
        new_error = 0.5 * (self.errors[q] + self.errors[f])
        self.errors = np.append(self.errors, new_error)

        # Reduce error of q and f
        self.errors[q] *= self.alpha
        self.errors[f] *= self.alpha

        # Update topology
        old_edge = tuple(sorted((q, f)))
        if old_edge in self.edges:
            self.edges.discard(old_edge)
            self.edge_ages.pop(old_edge, None)

        e1 = tuple(sorted((q, r)))
        e2 = tuple(sorted((f, r)))
        self.edges.add(e1)
        self.edges.add(e2)
        self.edge_ages[e1] = 0
        self.edge_ages[e2] = 0

    def quantization_error(self, data):
        dists = np.linalg.norm(data[:, None, :] - self.units[None, :, :], axis=2)
        return np.mean(np.min(dists, axis=1)**2)


# ========================== Visualization ==========================
def plot_gng(data, units, edges, title, subtitle=""):
    plt.figure(figsize=(9, 6.5))  # Slightly larger to prevent text overlap
    plt.scatter(data[:, 0], data[:, 1], c='lightgray', s=15, alpha=0.7, label='Data')
    for a, b in edges:
        plt.plot(units[[a, b], 0], units[[a, b], 1], 'b-', lw=1.2, alpha=0.6)
    plt.scatter(units[:, 0], units[:, 1], c='red', s=100, edgecolors='darkred',
                linewidth=2, zorder=10, label=f'{len(units)} neurons')
    # Combined title to avoid suptitle overlap issues
    full_title = f"{title}\n{subtitle}" if subtitle else title
    plt.title(full_title, fontsize=14, pad=25)  # Increased pad for breathing room
    plt.xlim(-1.5, 2.5)
    plt.ylim(-1.2, 1.5)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout(pad=2.0)  # Extra padding to prevent clipping
    plt.show()
    plt.close()  # Clean up to avoid memory issues


# ========================== Run ==========================
if __name__ == "__main__":
    np.random.seed(42)
    data, _ = make_moons(n_samples=1000, noise=0.08)

    gng = GrowingNeuralGas(
        epsilon_b=0.05,
        epsilon_n=0.0006,
        lambda_insert=100,
        max_age=88,
        alpha=0.5,
        beta=0.0005
    )

    print("Training Growing Neural Gas on Two Moons...")
    for i in range(12001):
        x = data[np.random.randint(0, len(data))]
        gng.train_one(x)

        if (i + 1) in [100, 1000, 3000, 8000, 12000]:
            print(f"Step {i+1:5d} → {len(gng.units):3d} neurons")
            plot_gng(data, gng.units, gng.edges,
                     f"Growing Neural Gas – Step {i+1:,}",
                     f"{len(gng.units)} neurons, {len(gng.edges)} edges")

    print("Final quantization error:", gng.quantization_error(data))