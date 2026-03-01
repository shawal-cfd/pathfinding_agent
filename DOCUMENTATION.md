# Dynamic Pathfinding Agent — Implementation & Analysis

## 1. Implementation Logic

### 1.1 A* Search Algorithm

**Evaluation Function:** \( f(n) = g(n) + h(n) \)

- **g(n)**: Actual path cost from start to node \( n \) (number of steps)
- **h(n)**: Heuristic estimate of cost from \( n \) to goal
- **f(n)**: Estimated total cost of path through \( n \)

**Implementation Details:**

1. **Data Structures:**
   - **Open Set**: Min-heap (priority queue) ordered by \( f(n) \). Nodes with lowest \( f \) are expanded first.
   - **g_score**: Dictionary mapping each node to its best known cost from start.
   - **came_from**: Dictionary for path reconstruction (parent pointers).

2. **Algorithm Flow:**
   - Initialize with start node: \( g(\text{start}) = 0 \), push \( (\text{f(start)}, \text{start}) \) into heap.
   - While the open set is not empty:
     - Pop the node with minimum \( f \).
     - If it is the goal, reconstruct and return the path.
     - For each unvisited neighbor:
       - Compute \( \text{tentative\_g} = g(\text{current}) + 1 \) (unit step cost).
       - If \( \text{tentative\_g} < g(\text{neighbor}) \), update \( g \), set parent, and push \( (f, \text{neighbor}) \) with \( f = g + h \).
   - If the open set is exhausted without reaching the goal, return no path.

3. **Optimality:** With an admissible heuristic (never overestimates), A* is optimal: it finds a shortest path if one exists.

---

### 1.2 Greedy Best-First Search (GBFS)

**Evaluation Function:** \( f(n) = h(n) \) only

- **h(n)**: Heuristic estimate of cost from \( n \) to goal
- No \( g(n) \): path cost from start is ignored

**Implementation Details:**

1. **Data Structures:**
   - **Open Set**: Min-heap ordered by \( h(n) \) only.
   - **visited**: Set of expanded nodes to avoid re-expansion.
   - **came_from**: Dictionary for path reconstruction.

2. **Algorithm Flow:**
   - Push \( (h(\text{start}), \text{start}) \) into the heap.
   - While the open set is not empty:
     - Pop the node with minimum \( h \).
     - Skip if already visited.
     - Mark as visited; if it is the goal, reconstruct and return the path.
     - For each unvisited neighbor, set parent and push \( (h(\text{neighbor}), \text{neighbor}) \).
   - If the open set is exhausted, return no path.

3. **Behavior:** GBFS expands nodes that appear closest to the goal by the heuristic, without considering how far they are from the start. It is fast but not guaranteed to find an optimal path.

---

### 1.3 Heuristic Functions

| Heuristic | Formula | Use Case |
|-----------|---------|----------|
| **Manhattan (L1)** | \( \|x_1 - x_2\| + \|y_1 - y_2\| \) | 4-connected grids (up/down/left/right); admissible and consistent |
| **Euclidean (L2)** | \( \sqrt{(x_1-x_2)^2 + (y_1-y_2)^2} \) | Diagonal-aware; admissible but can underestimate on 4-connected grids |
| **Chebyshev (L∞)** | \( \max(\|x_1-x_2\|, \|y_1-y_2\|) \) | 8-connected grids; admissible for 4-connected |

For 4-connected movement, Manhattan is typically the best choice: it matches the actual movement and keeps the heuristic admissible.

---

## 2. Pros & Cons (Based on Implementation & Experiments)

### 2.1 A* Search

**Pros:**
- **Optimal:** Finds a shortest path when using an admissible heuristic.
- **Complete:** Finds a path if one exists.
- **Efficient:** Expands fewer nodes than uninformed search by focusing on promising directions.
- **Flexible:** Works with different heuristics; Manhattan is well-suited for 4-connected grids.
- **Predictable:** Performance is stable across different map layouts.

**Cons:**
- **Memory:** Stores \( g \)-scores and parent pointers for many nodes.
- **Slower than GBFS:** More computation per node due to \( g(n) + h(n) \).
- **Heuristic-dependent:** Poor heuristics can lead to more expansions.

---

### 2.2 Greedy Best-First Search

**Pros:**
- **Fast:** Simple \( h(n) \) evaluation and fewer updates per node.
- **Low memory per node:** No \( g \)-score tracking.
- **Good in open spaces:** When the heuristic is accurate, it can reach the goal quickly.
- **Useful for real-time:** Suitable when speed matters more than optimality.

**Cons:**
- **Not optimal:** May return longer paths.
- **Deceptive heuristics:** Can explore many nodes when the heuristic misleads (e.g., obstacles between start and goal).
- **No path-cost awareness:** Ignores how far nodes are from the start.
- **Worst-case behavior:** Can degenerate to exploring a large portion of the grid.

---

### 2.3 Experimental Observations

| Scenario | A* | GBFS |
|----------|-----|------|
| **Empty grid, straight path** | Optimal path, moderate expansion | Often similar or fewer expansions |
| **Maze with obstacles** | Optimal path, focused expansion | Can explore many dead ends |
| **Deceptive layout** (e.g., wall near goal) | Still optimal, more expansions | Often many more expansions |
| **Dynamic re-planning** | More expensive per replan | Faster replans, but paths may be suboptimal |

---

## 3. Test Cases (Visual Proof)

Screenshots are generated by running:

```bash
python generate_test_screenshots.py
```

Output images are saved in `C:\Users\dell\pathfinding_agent\screenshots\`.

### 3.1 Best Case Scenario

**Setup:** Start and goal close, few or no obstacles between them. The heuristic points directly toward the goal.

- **A* Best Case:** Finds the optimal path with minimal expansion; visited nodes stay near the direct route.
- **GBFS Best Case:** Also expands few nodes; both algorithms perform similarly when the heuristic is accurate.

### 3.2 Worst Case Scenario

**Setup:** Maze-like layout or obstacles that mislead the heuristic. Goal may be “around the corner” from the heuristic’s perspective.

- **A* Worst Case:** Still finds the optimal path but expands more nodes; expansion remains focused on the optimal route.
- **GBFS Worst Case:** Can explore many dead ends and misleading directions; visited nodes often cover a large part of the grid before finding the goal.

---

## 4. File Reference

| File | Purpose |
|------|---------|
| `pathfinding.py` | Grid environment, A*, GBFS, heuristics |
| `main.py` | Pygame GUI, visualization, dynamic mode |
| `generate_test_screenshots.py` | Generates best/worst case screenshots for A* and GBFS |
