"""
Optimal Power Station Placement + TSP Inspection Route
-------------------------------------------------------
Industrial Engineering application:
1. Generate a hypothetical map with random demand points (e.g., neighborhoods/factories).
2. Find optimal power station locations using K-Means clustering
   (minimizing total distance between demand points and their nearest station).
3. Solve the Traveling Salesman Problem (TSP) to find the shortest
   inspection/maintenance route that visits all stations exactly once.
   - Construction: Nearest Neighbor heuristic
   - Improvement: 2-Opt local search
4. Visualize everything on the map with matplotlib.

Requirements:  pip install numpy matplotlib scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from itertools import combinations

# ----------------------------------------------------------------------
# 1. PARAMETERS
# ----------------------------------------------------------------------
RANDOM_SEED   = 42        # for reproducibility
N_DEMAND      = 120       # number of demand points on the map
N_STATIONS    = 8         # number of power stations to place
MAP_SIZE      = 100       # map is MAP_SIZE x MAP_SIZE (km)

rng = np.random.default_rng(RANDOM_SEED)

# ----------------------------------------------------------------------
# 2. GENERATE HYPOTHETICAL MAP (DEMAND POINTS)
# ----------------------------------------------------------------------
# Mix of clustered "urban" zones + uniform "rural" points for realism
urban_centers = rng.uniform(15, 85, size=(4, 2))
urban_points  = np.vstack([
    c + rng.normal(0, 6, size=(N_DEMAND // 6, 2)) for c in urban_centers
])
rural_points  = rng.uniform(0, MAP_SIZE, size=(N_DEMAND - len(urban_points), 2))
demand_points = np.clip(np.vstack([urban_points, rural_points]), 0, MAP_SIZE)

# Each demand point has a load (MW) — weight for the clustering
loads = rng.uniform(1, 10, size=len(demand_points))

# ----------------------------------------------------------------------
# 3. OPTIMAL STATION LOCATIONS (WEIGHTED K-MEANS)
# ----------------------------------------------------------------------
kmeans = KMeans(n_clusters=N_STATIONS, random_state=RANDOM_SEED, n_init=20)
kmeans.fit(demand_points, sample_weight=loads)
stations = kmeans.cluster_centers_          # optimal station coordinates
labels   = kmeans.labels_                   # which station serves each point

total_weighted_dist = sum(
    loads[i] * np.linalg.norm(demand_points[i] - stations[labels[i]])
    for i in range(len(demand_points))
)
print(f"Total weighted demand-to-station distance: {total_weighted_dist:,.1f} MW·km")

# Print optimal station coordinates
print("\nOptimal power station locations:")
print(f"{'Station':<10}{'X (km)':>10}{'Y (km)':>10}")
for idx, (x, y) in enumerate(stations):
    print(f"S{idx:<9}{x:>10.2f}{y:>10.2f}")

# ----------------------------------------------------------------------
# 4. TSP — DISTANCE MATRIX
# ----------------------------------------------------------------------
def distance_matrix(pts):
    """Euclidean distance matrix between all pairs of points."""
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))

D = distance_matrix(stations)

def tour_length(tour, D):
    """Total length of a closed tour."""
    return sum(D[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))

# ----------------------------------------------------------------------
# 5. TSP — NEAREST NEIGHBOR CONSTRUCTION
# ----------------------------------------------------------------------
def nearest_neighbor(D, start=0):
    n = len(D)
    unvisited = set(range(n)) - {start}
    tour = [start]
    while unvisited:
        last = tour[-1]
        nxt  = min(unvisited, key=lambda j: D[last, j])
        tour.append(nxt)
        unvisited.remove(nxt)
    return tour

# ----------------------------------------------------------------------
# 6. TSP — 2-OPT IMPROVEMENT
# ----------------------------------------------------------------------
def two_opt(tour, D):
    """Repeatedly reverse segments while it shortens the tour."""
    best = tour[:]
    improved = True
    while improved:
        improved = False
        for i, j in combinations(range(1, len(best)), 2):
            if j - i == 1:
                continue
            new = best[:i] + best[i:j][::-1] + best[j:]
            if tour_length(new, D) < tour_length(best, D) - 1e-10:
                best = new
                improved = True
    return best

nn_tour   = nearest_neighbor(D)
opt_tour  = two_opt(nn_tour, D)

print(f"Nearest-Neighbor tour length : {tour_length(nn_tour, D):.2f} km")
print(f"2-Opt improved tour length   : {tour_length(opt_tour, D):.2f} km")
print("Optimal visiting order (station indices):", opt_tour + [opt_tour[0]])

# ----------------------------------------------------------------------
# 7. VISUALIZATION
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 10))

# Demand points colored by their assigned station
scatter = ax.scatter(demand_points[:, 0], demand_points[:, 1],
                     c=labels, cmap="tab10", s=loads * 8,
                     alpha=0.55, edgecolors="none", label="Demand points")

# Lines from each demand point to its station (service assignment)
for i, p in enumerate(demand_points):
    s = stations[labels[i]]
    ax.plot([p[0], s[0]], [p[1], s[1]], color="gray", lw=0.3, alpha=0.4)

# Stations (optimal locations) — marked with red squares + name and coordinates
ax.scatter(stations[:, 0], stations[:, 1], marker="s", s=250,
           c="red", edgecolors="black", zorder=5, label="Power stations (optimal)")
for idx, (x, y) in enumerate(stations):
    ax.annotate(f"S{idx}\n({x:.1f}, {y:.1f})", (x, y),
                textcoords="offset points", xytext=(10, 8),
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow",
                          ec="black", lw=0.6, alpha=0.9))

# TSP route (closed loop)
route = opt_tour + [opt_tour[0]]
ax.plot(stations[route, 0], stations[route, 1],
        "b--", lw=2, zorder=4, label=f"TSP route ({tour_length(opt_tour, D):.1f} km)")

ax.set_title("Optimal Power Station Placement + TSP Maintenance Route")
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_xlim(0, MAP_SIZE)
ax.set_ylim(0, MAP_SIZE)
ax.legend(loc="upper right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("power_stations_map.png", dpi=150)
plt.show()
