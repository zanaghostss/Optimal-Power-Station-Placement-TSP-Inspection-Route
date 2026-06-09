# Mathematical Modeling: Optimal Power Station Placement + TSP Inspection Route

The problem is solved in two phases, exactly mirroring the structure of the Python code.

---

## Phase 1 — Optimal Station Location

### 1.A Continuous Model (Multi-Source Weber Problem — what K-Means approximates)

**Sets and parameters**

| Symbol | Meaning |
|---|---|
| I = {1, …, m} | set of demand points |
| J = {1, …, p} | set of stations to locate (p = N_STATIONS) |
| aᵢ ∈ ℝ² | coordinates of demand point i |
| wᵢ > 0 | load (weight) of demand point i, in MW |

**Decision variables**

- sⱼ ∈ ℝ² : coordinates of station j (continuous — anywhere on the map)
- zᵢⱼ ∈ {0, 1} : 1 if demand point i is assigned to station j, 0 otherwise

**Model**

```
min   Σ_{i∈I} Σ_{j∈J}  wᵢ · zᵢⱼ · ‖aᵢ − sⱼ‖

s.t.  Σ_{j∈J} zᵢⱼ = 1          ∀ i ∈ I      (each demand point served by exactly one station)
      zᵢⱼ ∈ {0, 1}             ∀ i, j
      sⱼ ∈ [0, L]²             ∀ j ∈ J      (stations must lie inside the L×L map)
```

> Note: K-Means minimizes the weighted **squared** distance Σ wᵢ‖aᵢ − sⱼ‖²,
> which is a tractable approximation of this model. Each cluster centroid
> sⱼ* = Σᵢ wᵢzᵢⱼaᵢ / Σᵢ wᵢzᵢⱼ is the optimal location for its assigned demand.

### 1.B Discrete Alternative (p-Median) — if stations may only be built at candidate sites

Let K be a finite set of candidate sites with dᵢₖ the distance from demand i to site k.

- yₖ ∈ {0,1}: 1 if a station is built at site k
- zᵢₖ ∈ {0,1}: 1 if demand i is assigned to site k

```
min   Σ_{i∈I} Σ_{k∈K}  wᵢ · dᵢₖ · zᵢₖ

s.t.  Σ_{k∈K} zᵢₖ = 1          ∀ i ∈ I      (full assignment)
      zᵢₖ ≤ yₖ                 ∀ i, k       (assign only to open stations)
      Σ_{k∈K} yₖ = p                        (exactly p stations are opened)
      yₖ, zᵢₖ ∈ {0, 1}
```

---

## Phase 2 — TSP Inspection Route (Miller–Tucker–Zemlin formulation)

After Phase 1, station coordinates s₁,…,sₚ are fixed.
Let n = p and dᵢⱼ = ‖sᵢ − sⱼ‖ (Euclidean distance matrix used in the code).

**Decision variables**

- xᵢⱼ ∈ {0,1}: 1 if the route travels directly from station i to station j
- uᵢ ∈ ℝ: auxiliary order variable of station i in the tour (MTZ)

**Model**

```
min   Σ_{i=1}^{n} Σ_{j≠i}  dᵢⱼ · xᵢⱼ

s.t.  Σ_{j≠i} xᵢⱼ = 1                      ∀ i            (leave every station exactly once)
      Σ_{i≠j} xᵢⱼ = 1                      ∀ j            (enter every station exactly once)
      uᵢ − uⱼ + n·xᵢⱼ ≤ n − 1              ∀ i,j ∈ {2,…,n}, i≠j   (MTZ subtour elimination)
      1 ≤ uᵢ ≤ n − 1                       ∀ i ∈ {2,…,n}
      xᵢⱼ ∈ {0, 1}                          ∀ i ≠ j
```

**Alternative subtour elimination (DFJ)** — exponentially many but tighter constraints:

```
Σ_{i∈S} Σ_{j∈S, j≠i} xᵢⱼ ≤ |S| − 1        ∀ S ⊂ {1,…,n}, 2 ≤ |S| ≤ n−1
```

---

## Mapping Model → Code

| Model element | Code element |
|---|---|
| min Σ wᵢ‖aᵢ − sⱼ‖² (Phase 1.A, squared) | `KMeans(...).fit(demand_points, sample_weight=loads)` |
| assignment zᵢⱼ | `labels = kmeans.labels_` |
| station locations sⱼ | `stations = kmeans.cluster_centers_` |
| distance matrix dᵢⱼ | `D = distance_matrix(stations)` |
| TSP objective Σ dᵢⱼxᵢⱼ | `tour_length(tour, D)` |
| heuristic for x (construction) | `nearest_neighbor(D)` |
| heuristic for x (improvement) | `two_opt(tour, D)` |

> The code solves TSP **heuristically** (NN + 2-Opt gives a good feasible solution,
> not a proven optimum). For an exact solution of the MTZ model, the same
> formulation can be fed to a MIP solver such as PuLP/CBC, Gurobi, or CPLEX.

## Complexity Notes

- p-Median and TSP are both **NP-hard**.
- Nearest Neighbor: O(n²). 2-Opt: O(n²) per pass until local optimum.
- K-Means (Lloyd's algorithm): O(m·p·t) per run, t = iterations; converges to a local optimum, hence `n_init=20` restarts in the code.
