

# **A Student's Guide to Graph Theory: From Königsberg to Kruskal's**

Welcome to the fascinating world of graph theory\! This module will introduce you to graphs, a fundamental concept in computer science and discrete mathematics. We'll explore what they are, why they are so incredibly powerful, and how we can use algorithms to unlock their secrets.

## **Introduction: What are Graphs and Why Do They Matter?**

Our story begins in the 18th century in the city of Königsberg, Prussia (now Kaliningrad, Russia). The city was built around a river, with seven bridges connecting two large islands and the mainland. A famous puzzle of the day asked: could a person walk through the city, cross each of the seven bridges exactly once, and return to their starting point?.1

In 1735, the great mathematician Leonhard Euler solved this puzzle. He realized that the specific layout of the city—the size of the islands, the length of the bridges—was irrelevant. The only thing that mattered was the pattern of connections. He abstracted the problem by representing each landmass as a point (a **vertex** or **node**) and each bridge as a line connecting two points (an **edge**). By simplifying the complex map into this abstract structure, which he called a graph, he was able to prove that no such path was possible.1

This act of abstraction was the birth of graph theory, and it is the very essence of its power. The core idea is that many complex, real-world problems, which at first glance seem unrelated, share an underlying structure of "things" and "connections." A logistics problem is about delivery points and the routes between them.3 A social network is about people and their friendships.4 The World Wide Web is about pages and the hyperlinks that connect them.5 By modeling all these problems using the universal language of graphs, we gain access to a powerful, pre-existing toolkit of algorithms that can solve them.

Learning an algorithm like Dijkstra's for finding the shortest path doesn't just teach you how to build a GPS. It gives you a method for finding the "cheapest" or "most efficient" path in *any* system that can be modeled as a weighted graph, whether that's minimizing latency in a computer network or analyzing interactions between proteins in a cell.1

Formally, a graph G is defined as an ordered pair G=(V,E), where V is a set of vertices and E is a set of edges, which are pairs of vertices.7 Throughout this tutorial, we will explore how to represent these simple structures and the powerful algorithms that operate on them.

## **The Language of Graphs: Core Concepts and Representations**

Before we can run algorithms, we need a clear vocabulary to describe the different kinds of graphs and their properties.

### **A Graph Taxonomy: The Cast of Characters**

Graphs come in several flavors, each suited to modeling different kinds of relationships.7

* **Directed vs. Undirected Graphs**: An **undirected graph** is like a two-way street. If an edge connects vertex A and vertex B, you can travel from A to B and from B to A. This is perfect for modeling things like a Facebook friendship, which is mutual. A **directed graph** (or **digraph**) is like a one-way street. An edge from A to B does not imply an edge from B to A. This is ideal for modeling a Twitter follow, where you can follow someone who doesn't follow you back.8  
* **Weighted vs. Unweighted Graphs**: In an **unweighted graph**, every edge is the same. The only thing that matters is whether a connection exists. In a **weighted graph**, each edge is assigned a numerical value, or "weight." This weight can represent distance, cost, time, or any other measurable quantity. Weighted graphs are essential for optimization problems, like finding the shortest route between two cities.1  
* **Simple Graphs vs. Multigraphs**: A **simple graph** has at most one edge between any pair of vertices and no **loops** (edges that connect a vertex to itself). A **multigraph** allows for multiple edges between the same two vertices. The Königsberg bridge problem, with its two pairs of parallel bridges, is a classic example of a multigraph.1  
* **Essential Terminology**:  
  * **Degree**: The degree of a vertex is the number of edges connected to it. In a directed graph, we distinguish between *in-degree* (edges pointing to the vertex) and *out-degree* (edges pointing away).1  
  * **Walk vs. Path**: A **walk** is a sequence of vertices where each adjacent pair in the sequence is connected by an edge. A walk can repeat vertices and edges. A **path** is a walk where no vertices (and thus no edges) are repeated.1 For example, in a graph with edges (A,B), (B,C), (C,A), the sequence A-B-C-A is a walk, but A-B-C is a path.  
  * **Cycle**: A **cycle** is a path that starts and ends at the same vertex.1  
  * **Connectivity**: A graph is **connected** if there is a path between every pair of vertices. If not, it is **disconnected**. A disconnected graph is made up of several **connected components**.1

### **Representing Graphs: From Drawing to Data Structure**

To work with graphs on a computer, we need to store them in a data structure. The choice of representation involves trade-offs between memory usage and the efficiency of common operations.1

* **Edge List**: The simplest representation. It's just a list of pairs (or tuples) representing the edges. For a graph with vertices {A, B, C, D} and edges connecting A-B, A-C, B-C, and C-D, the edge list would be {{A,B}, {A,C}, {B,C}, {C,D}}. While simple and memory-efficient for sparse graphs, finding all neighbors of a specific vertex requires scanning the entire list, which is slow.1  
* **Adjacency Matrix**: An N×N matrix (where N is the number of vertices), which we'll call A. The entry Ai,j​ is 1 if there is an edge between vertex i and vertex j, and 0 otherwise. For weighted graphs, the entry can store the edge's weight, with ∞ representing no edge. This allows for very fast O(1) lookup to check if an edge exists. However, it requires O(N2) space, which is very inefficient for **sparse graphs** (graphs with far fewer edges than the maximum possible).1  
* **Adjacency List (Adjacency Structure)**: This is often the best of both worlds. It is an array (or map) of lists. For each vertex i, the list at index i contains all of its neighbors. This representation is space-efficient for sparse graphs, using O(V+E) space (where V is the number of vertices and E is the number of edges), and it's fast to iterate over the neighbors of a given vertex.1  
* **Incidence Matrix**: An M×N matrix (where M is the number of edges), which we'll call I. The entry Ii,j​ is 1 if edge i is connected to vertex j. This representation is less common but has applications in fields like electrical engineering and numerical analysis.1

The following table summarizes the trade-offs for the most common representations.

| Representation | Space Complexity | Time to Check Edge (u,v) | Time to Find Neighbors(v) |
| :---- | :---- | :---- | :---- |
| **Adjacency List** | O(V+E) | O(degree(u)) | O(degree(v)) |
| **Adjacency Matrix** | O(V2) | O(1) | O(V) |
| **Edge List** | O(E) | O(E) | O(E) |

### **A Mathematical Shortcut: Counting Walks with Matrices**

One of the most elegant results in graph theory connects it to linear algebra. If A is the adjacency matrix of a graph, then the entry (i,j) of the matrix Ak (A raised to the power of k) gives the exact number of different walks of length k from vertex i to vertex j.1

**Example:** Consider this simple graph and its adjacency matrix A:

* Vertices: 1, 2, 3  
* Edges: (1,2), (1,3), (2,3)

A=​011​101​110​​  
Let's find the number of walks of length 2 from vertex 1 to vertex 2\. We compute A2:

A2=A×A=​011​101​110​​​011​101​110​​=​211​121​112​​  
The entry (1,2) of A2 is 1\. This means there is exactly one walk of length 2 from vertex 1 to vertex 2\. Let's trace it: 1-3-2. This powerful property allows us to answer complex counting questions using standard matrix multiplication.

---

## **Self-Check: Why is the adjacency matrix of an undirected graph always symmetric? What does an entry on the main diagonal (e.g., Ai,i​) represent?**

## **Exploring the Maze: Graph Traversal Algorithms**

A fundamental task in graph algorithms is **traversal**: a systematic process for visiting every vertex and exploring every edge. This is the building block for many other algorithms. Think of a web crawler indexing the internet by following hyperlinks, or a program mapping all devices connected to a network.10 The two primary traversal strategies are Depth-First Search (DFS) and Breadth-First Search (BFS).

### **Depth-First Search (DFS): Going Deep**

DFS explores a graph by going as deep as possible down one path before backtracking.

* **Analogy**: The best way to think of DFS is solving a maze.1 You follow a single path, marking your route. When you hit a dead end, you backtrack to the most recent junction where you had an unexplored path and try that one. You repeat this until you've explored all paths from all junctions.  
* **Mechanism**: DFS is naturally implemented using a **stack**, which follows a Last-In, First-Out (LIFO) principle. This can be the program's own call stack (in a recursive implementation) or an explicit stack data structure (in an iterative one).1 The algorithm works as follows:  
  1. Start at a chosen vertex, mark it as visited, and push it onto the stack.  
  2. While the stack is not empty, pop a vertex u.  
  3. For each unvisited neighbor v of u:  
     * Mark v as visited.  
     * Push v onto the stack.  
* **Applications**: Because of its deep-diving nature, DFS is excellent for:  
  * **Cycle Detection**: If DFS encounters a vertex that is already in the current recursion stack (an ancestor), it has found a cycle.10  
  * **Path Finding**: It can find a path between two nodes (though not necessarily the shortest one).  
  * **Topological Sorting**: For a Directed Acyclic Graph (DAG), DFS can produce a linear ordering of vertices such that for every directed edge from u to v, u comes before v. This is crucial for scheduling tasks with dependencies.14

### **Breadth-First Search (BFS): Exploring Wide**

BFS takes the opposite approach. It explores the graph layer by layer, visiting all of a vertex's immediate neighbors before moving on to their neighbors.

* **Analogy**: Imagine dropping a stone in a pond. BFS is like the ripple effect: it explores the vertices closest to the start, then the next closest, and so on, expanding outward in concentric circles.16  
* **Mechanism**: The key to BFS's level-by-level traversal is the **queue**, a First-In, First-Out (FIFO) data structure.16  
  1. Start at a chosen vertex, mark it as visited, and add it to the queue.  
  2. While the queue is not empty, remove the vertex u from the front of the queue.  
  3. For each unvisited neighbor v of u:  
     * Mark v as visited.  
     * Add v to the back of the queue.  
* **Key Property**: BFS has a critically important property: for **unweighted graphs**, it is guaranteed to find the shortest path (in terms of the number of edges) from the starting vertex to all other vertices.11 This is because it always explores all paths of length  
  k before exploring any path of length k+1.

The table below provides a strategic comparison to help you choose the right traversal algorithm for a given task.

| Feature | Depth-First Search (DFS) | Breadth-First Search (BFS) |
| :---- | :---- | :---- |
| **Data Structure** | Stack (LIFO) or Recursion | Queue (FIFO) |
| **Search Order** | Goes deep down one path before backtracking | Explores level by level, wide before deep |
| **Path Finding** | Finds a path, but not guaranteed to be the shortest | Guarantees shortest path (in edge count) for unweighted graphs |
| **Memory Usage** | Can be more memory-efficient for "wide" graphs | Can use more memory for "deep" graphs as the queue can grow large |
| **Common Use Cases** | Cycle detection, topological sorting, solving mazes | Shortest path in unweighted graphs, network broadcasting, web crawling |

---

## **Self-Check: If you were building a feature to find the shortest chain of "friend of a friend" connections between two people on a social network, would you use DFS or BFS? Why?**

## **Finding the Best Route: Shortest Path Algorithms**

While BFS is perfect for finding the path with the fewest edges, many real-world problems involve weighted graphs. The question is no longer "What's the path with the fewest stops?" but "What's the path with the minimum total distance, cost, or time?".1 This is the

**shortest path problem**, and it is at the heart of applications like:

* **GPS Navigation**: Google Maps and Waze use shortest path algorithms to find the fastest route, where edge weights can be travel time, distance, or even a combination that accounts for traffic.4  
* **Network Routing**: Internet routers use protocols like OSPF (Open Shortest Path First) to find the most efficient path to send data packets, minimizing latency. The "cost" of an edge might be inversely related to bandwidth.17  
* **Logistics and Supply Chains**: Companies optimize delivery routes to minimize fuel costs and time.20

### **Dijkstra's Algorithm: A Greedy Approach**

The most famous algorithm for solving the single-source shortest path problem on weighted graphs with non-negative edge weights is Dijkstra's Algorithm, developed by Edsger Dijkstra in 1956\.1

* **Core Idea**: Dijkstra's algorithm is a **greedy algorithm**. At each step, it makes a locally optimal choice in the hope of finding a globally optimal solution.1 It works like an enhanced version of BFS. Instead of a simple queue that treats all neighbors equally, it uses a  
  **priority queue** to always explore the "closest" unvisited vertex from the source.18  
* **The Algorithm Step-by-Step**:  
  1. **Initialization**: Create a distance array, Dist, and initialize the distance to the start node A as 0 and all other nodes as ∞. Create a set of visited nodes, initially empty.  
  2. Main Loop: While there are unvisited nodes:  
     a. Select the unvisited node P with the smallest distance in Dist. This is the greedy choice.  
     b. Mark P as visited.  
     c. For each neighbor X of P, perform a "relaxation" step: calculate the distance to X through P. This is Dist(P) \+ Length(P, X). If this new distance is less than the current Dist(X), update Dist(X) with this shorter path's distance.  
  3. **Termination**: The algorithm terminates when all reachable nodes have been visited. The Dist array now contains the shortest path distances from A to every other node.

The greedy choice at the core of Dijkstra's algorithm—always selecting the unvisited node with the smallest known distance—is not just a good heuristic; it is provably correct. Once the algorithm selects a node P and marks it as visited, the distance calculated for P, Dist(P), is guaranteed to be the absolute shortest path from the source. Any other potential path to P would have to pass through another unvisited node Q. However, since we always pick the node with the minimum distance, the distance to Q must be greater than or equal to the distance to P. Because all edge weights are non-negative, the path through Q could never be shorter than the path we have already found for P. This guarantee is the magic of the algorithm, but it is also why it fails if negative edge weights are present, as they can violate this fundamental assumption.1

**Worked Example:** Let's trace Dijkstra's algorithm to find the shortest path from A to F on the graph below.1

\!(https://i.imgur.com/example-graph.png)  
A sample weighted graph for tracing Dijkstra's algorithm.  
We will track the Dist array and the Visited set at each step. P is the current node being processed.

| Step | P | Visited | Dist(A) | Dist(B) | Dist(C) | Dist(D) | Dist(E) | Dist(F) | Notes |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **0** | \- | {} | 0 | ∞ | ∞ | ∞ | ∞ | ∞ | Initialization. |
| **1** | A | {A} | 0 | 40 | 15 | ∞ | ∞ | ∞ | Visit A. Relax neighbors B (0+40) and C (0+15). |
| **2** | C | {A, C} | 0 | 35 | 15 | 115 | ∞ | ∞ | Next closest is C (15). Relax B: min(40, 15+20)=35. Relax D: 15+100=115. |
| **3** | B | {A, C, B} | 0 | 35 | 15 | 45 | 60 | 41 | Next closest is B (35). Relax D: min(115, 35+10)=45. Relax E: 35+25=60. Relax F: 35+6=41. |
| **4** | F | {A, C, B, F} | 0 | 35 | 15 | 45 | 49 | 41 | Next closest is F (41). Relax E: min(60, 41+8)=49. F is our target, so we can stop. |

The shortest distance from A to F is 41\. If we continued, the algorithm would find the shortest paths to all other nodes as well.

* **Implementation and Complexity**: A naive implementation that scans the distance array at each step to find the minimum takes O(V2) time. By using a binary heap as a priority queue, the time complexity is optimized to O((V+E)logV), which is much faster for sparse graphs.17

## **Building the Most Efficient Network: Minimum Spanning Trees**

Imagine you need to connect a set of houses to a utility service (like water or internet). You can lay pipes or cables between any two houses, but each connection has a cost. How do you connect all the houses with the minimum possible total cost? This is the **Minimum Spanning Tree (MST)** problem.

First, let's define our terms 1:

* A **tree** is a connected graph with no cycles.  
* A **spanning tree** of a connected graph G is a subgraph that includes all of G's vertices and is a tree. A spanning tree for a graph with N vertices will always have exactly N−1 edges.  
* A **Minimum Spanning Tree (MST)** is a spanning tree of a weighted, undirected graph whose edges sum to the minimum possible weight.

MSTs are fundamental in network design, including 25:

* **Telecommunications**: Laying fiber optic cables to connect cities.  
* **Infrastructure**: Designing electrical grids or water pipeline networks.  
* **Data Science**: Used in cluster analysis algorithms to group similar data points.

### **Kruskal's Algorithm: Another Greedy Approach**

Kruskal's algorithm is a simple and elegant greedy algorithm for finding an MST.1

* **Core Idea**: The strategy is beautifully simple: consider all edges in the entire graph, sorted by weight from smallest to largest. Add each edge to your spanning tree if and only if it does not form a cycle with the edges you've already added.  
* **The Cycle Detection Problem**: The crucial step is efficiently checking if adding an edge (u, v) creates a cycle. This is equivalent to asking: "Are vertices u and v already connected in the tree we are building?" The canonical data structure for this task is the **Disjoint Set Union (DSU)**, also known as Union-Find.28  
  * **DSU Explained**: Imagine each vertex starts in its own separate set. The DSU structure supports two main operations:  
    1. find(v): Returns an identifier for the set that vertex v belongs to.  
    2. union(u, v): Merges the sets containing u and v into a single set.  
  * **Using DSU in Kruskal's**: For each edge (u, v) in our sorted list, we check if find(u) is the same as find(v). If they are the same, u and v are already in the same connected component, so adding the edge (u, v) would create a cycle. We discard it. If they are different, the edge connects two previously disconnected components. We add it to our MST and perform union(u, v).  
* **The Algorithm Step-by-Step**:  
  1. Create a list of all edges in the graph.  
  2. Sort the list of edges in non-decreasing order of their weights.  
  3. Initialize an empty MST. Initialize a DSU structure with each vertex in its own set.  
  4. Iterate through the sorted edges. For each edge (u, v):  
     * If find(u) is not equal to find(v):  
       * Add the edge (u, v) to the MST.  
       * Perform union(u, v).  
  5. Stop when the MST has N−1 edges.

It's insightful to compare the greedy strategies of Kruskal's algorithm and Prim's algorithm (which is mechanistically very similar to Dijkstra's). Prim's algorithm is "vertex-centric"—it grows a single tree from a starting vertex, always adding the cheapest edge that connects a vertex *inside* the tree to one *outside*. It's a "conquer from within" approach. Kruskal's, on the other hand, is "edge-centric." It doesn't care about a single growing component. It looks at the cheapest edge *anywhere* in the graph and uses it to connect whatever two components it happens to bridge. This creates a "forest" of trees that eventually merge into one.30 The remarkable result is that both of these fundamentally different local strategies are guaranteed to arrive at the same global optimum—the MST.

---

## **Self-Check: Does Kruskal's algorithm work on disconnected graphs? If so, what is the result?**

## **Key Takeaways**

* **Graphs are a universal language** for modeling problems based on relationships and connections. Abstracting a problem into a graph unlocks a powerful toolkit of algorithms.  
* **Graph representations** (Adjacency List, Adjacency Matrix) have critical trade-offs in space and time efficiency. The Adjacency List is usually the best choice for sparse graphs common in the real world.  
* **Graph traversal** is a fundamental operation. **DFS** (using a stack) goes deep and is useful for pathfinding and cycle detection. **BFS** (using a queue) explores layer-by-layer and finds the shortest path in unweighted graphs.  
* **Dijkstra's Algorithm** is the classic greedy solution for the single-source shortest path problem in weighted graphs with non-negative edges. It uses a priority queue to always explore the "closest" node.  
* **Minimum Spanning Trees (MSTs)** are used to find the cheapest way to connect all vertices in a weighted graph. **Kruskal's Algorithm** is a greedy method that sorts all edges and adds the cheapest ones that do not form a cycle, using a Disjoint Set Union data structure for efficient cycle detection.

## **Practice Problems**

Here are some problems to test your understanding.

1. **Conceptual:** You are designing a social network. When a user views another's profile, you want to show their "degree of separation" (e.g., "friend," "friend of a friend," etc.). Which traversal algorithm (DFS or BFS) would you use to compute this, and why is it the correct choice?  
2. **Algorithm Tracing (Dijkstra's):** Consider the graph from the Dijkstra's example. Find the shortest path distances from vertex **C** to all other vertices by tracing Dijkstra's algorithm. Show your work in a table similar to the one in the tutorial.  
3. **Algorithm Tracing (Kruskal's):** Find the Minimum Spanning Tree for the following graph by tracing Kruskal's algorithm. List the edges in the order they are considered and state whether each is added or rejected.  
   * Vertices: A, B, C, D, E  
   * Edges: (A,B,1), (B,C,2), (C,D,3), (D,E,4), (E,A,5), (A,C,6), (B,D,7)  
4. **Short Proof:** Prove that any tree with N vertices must have exactly N−1 edges. (Hint: Use induction or consider the process of building a tree by adding vertices).  
5. **Conceptual:** Explain why Kruskal's algorithm is considered "greedy." Then, explain why Prim's algorithm (which grows one tree from a start node by always adding the cheapest edge connecting the tree to a non-tree vertex) is also considered "greedy," even though its process is different.

### **Solutions and Hints**

1. **Solution:** You would use **Breadth-First Search (BFS)**. BFS explores the graph layer by layer from the starting user. All direct friends (distance 1\) will be found first, then all friends-of-friends (distance 2), and so on. The first time the target user is found, BFS guarantees it's via the shortest path in terms of connections. DFS would find a path, but it might be a very long, convoluted one.  
2. **Hint:** Start with Dist(C) \= 0 and all others as ∞. The first node to visit is C. Its neighbors are A, B, and D. Update their distances and proceed.  
3. **Hint:** First, sort the edges by weight: (A,B,1), (B,C,2), (C,D,3), (D,E,4), (E,A,5), (A,C,6), (B,D,7). Start adding them and use a DSU or draw the components to check for cycles. The edge (A,C,6) will be the first one you reject.  
4. **Hint (Inductive Proof):**  
   * **Base Case:** A tree with N=1 vertex has 0 edges. 1−1=0. True.  
   * **Inductive Hypothesis:** Assume any tree with k vertices has k−1 edges.  
   * **Inductive Step:** Consider a tree with k+1 vertices. Remove a leaf node (a node with degree 1\) and its connecting edge. The remaining graph is still a tree and has k vertices. By the hypothesis, it has k−1 edges. When you add the leaf node and its edge back, you now have k+1 vertices and (k−1)+1=k edges.  
5. **Solution:** Kruskal's is greedy because at each step, it makes the locally optimal choice of picking the absolute cheapest available edge in the entire graph without looking ahead to see if this choice will lead to a more expensive graph later. Prim's is greedy because at each step, it makes the locally optimal choice of picking the cheapest edge to expand its *current* tree, without considering whether a different, slightly more expensive expansion now might open up much cheaper options later. Both make the "best-looking" choice at the moment, just from different perspectives (global edge vs. local expansion).

## **Extension Questions**

1. **Negative Weights:** Dijkstra's algorithm fails with negative edge weights. Draw a small graph (3 or 4 vertices) with at least one negative edge. Choose a start and end vertex and show the path Dijkstra's algorithm finds versus the true shortest path. Explain precisely where the algorithm's greedy assumption was violated.  
2. **The Traveling Salesman Problem (TSP):** The TSP asks for the shortest tour that visits every city exactly once. For N cities, a brute-force approach checks all (2(N−1)\!​) possible tours. If a supercomputer can check one trillion (1012) tours per second, what is the largest number of cities (N) for which it could solve the TSP by brute force in under a year? (A year has approx. 3.15×107 seconds). This illustrates why TSP is considered an "intractable" problem.

## **Extra Learning Resources**

### **YouTube Videos**

1. **Dijkstra's Shortest Path Algorithm | A Step-by-Step Guide by MisterCode**  
   * **Link:**([https://www.youtube.com/watch?v=4xROtuo1xAw](https://www.youtube.com/watch?v=4xROtuo1xAw))  
   * **Why it's good:** An excellent, clear, and concise step-by-step walkthrough of Dijkstra's algorithm. It covers the pseudocode, a worked example, and complexity analysis, making it perfect for reinforcing the lecture material.  
2. **Kruskal's Algorithm for Minimum Spanning Trees by CodeShot**  
   * **Link:**([https://www.youtube.com/watch?v=OxfTT8slSLs](https://www.youtube.com/watch?v=OxfTT8slSLs))  
   * **Why it's good:** This video provides a great visual explanation of Kruskal's algorithm and clearly explains the role of the Union-Find (Disjoint Set) data structure in detecting cycles, which is a crucial implementation detail.

### **Online Articles & Visualizations**

1. **GeeksForGeeks: Applications of Depth First Search**  
   * **Link:** [https://www.geeksforgeeks.org/dsa/applications-of-depth-first-search/](https://www.geeksforgeeks.org/dsa/applications-of-depth-first-search/)  
   * **Why it's good:** A comprehensive list of where DFS is used in the real world, from finding cycles and connected components to solving puzzles like Sudoku. This helps connect the abstract algorithm to concrete applications.  
2. **VisuAlgo \- Graph Traversal**  
   * **Link:** [https://visualgo.net/en/graphds](https://visualgo.net/en/graphds)  
   * **Why it's good:** This is an interactive visualization tool. You can create your own graphs or use their examples and watch algorithms like BFS, DFS, Dijkstra's, and Kruskal's run step-by-step. Actively playing with the algorithms is one of the best ways to build a deep, intuitive understanding.

#### **Works cited**

1. 3\_P1\_Graphs\_handout.pdf  
2. Graph theory | Problems & Applications | Britannica, accessed August 15, 2025, [https://www.britannica.com/topic/graph-theory](https://www.britannica.com/topic/graph-theory)  
3. Algorithms for the Travelling Salesman Problem \- Routific, accessed August 15, 2025, [https://www.routific.com/blog/travelling-salesman-problem](https://www.routific.com/blog/travelling-salesman-problem)  
4. Real-World Applications of Graph Algorithms: Powering Modern Technology \- AlgoCademy, accessed August 15, 2025, [https://algocademy.com/blog/real-world-applications-of-graph-algorithms-powering-modern-technology/](https://algocademy.com/blog/real-world-applications-of-graph-algorithms-powering-modern-technology/)  
5. Applications of Graphs in Data Structures \- PrepBytes, accessed August 15, 2025, [https://www.prepbytes.com/blog/graphs/applications-of-graphs-in-data-structures/](https://www.prepbytes.com/blog/graphs/applications-of-graphs-in-data-structures/)  
6. Travelling salesman problem \- Wikipedia, accessed August 15, 2025, [https://en.wikipedia.org/wiki/Travelling\_salesman\_problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem)  
7. Graph Theory Fundamentals for Beginners: An Easy Guide, accessed August 15, 2025, [https://www.numberanalytics.com/blog/graph-theory-fundamentals-beginners-guide](https://www.numberanalytics.com/blog/graph-theory-fundamentals-beginners-guide)  
8. A Gentle Introduction To Graph Theory | by Vaidehi Joshi | basecs \- Medium, accessed August 15, 2025, [https://medium.com/basecs/a-gentle-introduction-to-graph-theory-77969829ead8](https://medium.com/basecs/a-gentle-introduction-to-graph-theory-77969829ead8)  
9. 15.1 Introduction to Graph Theory, accessed August 15, 2025, [https://mathbooks.unl.edu/Contemporary/sec-graph-intro.html](https://mathbooks.unl.edu/Contemporary/sec-graph-intro.html)  
10. Applications, Advantages and Disadvantages of Depth First Search (DFS) \- GeeksforGeeks, accessed August 15, 2025, [https://www.geeksforgeeks.org/dsa/applications-of-depth-first-search/](https://www.geeksforgeeks.org/dsa/applications-of-depth-first-search/)  
11. Graph Traversal Algorithms Explained: DFS, BFS & Applications \- PuppyGraph, accessed August 15, 2025, [https://www.puppygraph.com/blog/graph-traversal](https://www.puppygraph.com/blog/graph-traversal)  
12. Depth-First Search (DFS) | Brilliant Math & Science Wiki, accessed August 15, 2025, [https://brilliant.org/wiki/depth-first-search-dfs/](https://brilliant.org/wiki/depth-first-search-dfs/)  
13. Depth-first search \- Wikipedia, accessed August 15, 2025, [https://en.wikipedia.org/wiki/Depth-first\_search](https://en.wikipedia.org/wiki/Depth-first_search)  
14. Depth First Search Applications \- Tech Sauce \- Medium, accessed August 15, 2025, [https://techsauce.medium.com/depth-first-search-applications-da529f59a4b2](https://techsauce.medium.com/depth-first-search-applications-da529f59a4b2)  
15. Mastering Depth-First Search in Computer Science \- Number Analytics, accessed August 15, 2025, [https://www.numberanalytics.com/blog/mastering-depth-first-search](https://www.numberanalytics.com/blog/mastering-depth-first-search)  
16. CS 225 | BFS & DFS, accessed August 15, 2025, [https://courses.grainger.illinois.edu/cs225/sp2021/resources/bfs-dfs/](https://courses.grainger.illinois.edu/cs225/sp2021/resources/bfs-dfs/)  
17. Dijkstra's Algorithm Explained: Comprehensive Guide to Shortest Paths \- Upper Route Planner, accessed August 15, 2025, [https://www.upperinc.com/glossary/route-optimization/dijkstras-algorithm/](https://www.upperinc.com/glossary/route-optimization/dijkstras-algorithm/)  
18. What is DSA Dijkstra's Algorithm used for in real life | Mbloging, accessed August 15, 2025, [https://www.mbloging.com/post/dijkstras-algorithm-real-life-uses](https://www.mbloging.com/post/dijkstras-algorithm-real-life-uses)  
19. Dijkstra'S Algorithms \- Meegle, accessed August 15, 2025, [https://www.meegle.com/en\_us/topics/algorithm/dijkstra's-algorithms](https://www.meegle.com/en_us/topics/algorithm/dijkstra's-algorithms)  
20. www.numberanalytics.com, accessed August 15, 2025, [https://www.numberanalytics.com/blog/dijkstras-algorithm-in-depth\#:\~:text=Dijkstra's%20algorithm%20can%20be%20used,costs%20and%20improve%20delivery%20times.](https://www.numberanalytics.com/blog/dijkstras-algorithm-in-depth#:~:text=Dijkstra's%20algorithm%20can%20be%20used,costs%20and%20improve%20delivery%20times.)  
21. Determination of the Fastest Path on Logistics Distribution by Using Dijkstra Algorithm \- Atlantis Press, accessed August 15, 2025, [https://www.atlantis-press.com/article/125963778.pdf](https://www.atlantis-press.com/article/125963778.pdf)  
22. New Method Is the Fastest Way To Find the Best Routes | Quanta Magazine, accessed August 15, 2025, [https://www.quantamagazine.org/new-method-is-the-fastest-way-to-find-the-best-routes-20250806/](https://www.quantamagazine.org/new-method-is-the-fastest-way-to-find-the-best-routes-20250806/)  
23. Dijkstra's Algorithm: A Deep Dive \- Number Analytics, accessed August 15, 2025, [https://www.numberanalytics.com/blog/dijkstras-algorithm-deep-dive](https://www.numberanalytics.com/blog/dijkstras-algorithm-deep-dive)  
24. How Dijkstra's Algorithm Works \- YouTube, accessed August 15, 2025, [https://www.youtube.com/watch?v=EFg3u\_E6eHU](https://www.youtube.com/watch?v=EFg3u_E6eHU)  
25. Understanding Kruskal's Algorithm in Design and Analysis of Algorithms \- Pass4sure, accessed August 15, 2025, [https://www.pass4sure.com/blog/understanding-kruskals-algorithm-in-design-and-analysis-of-algorithms/](https://www.pass4sure.com/blog/understanding-kruskals-algorithm-in-design-and-analysis-of-algorithms/)  
26. Kruskal's Algorithm: Bridging Networks with Simplicity \- Great Learning, accessed August 15, 2025, [https://www.mygreatlearning.com/blog/kruskals-algorithm/](https://www.mygreatlearning.com/blog/kruskals-algorithm/)  
27. Applications Of Kruskal's Algorithm \- HeyCoach, accessed August 15, 2025, [https://heycoach.in/blog/applications-of-kruskals-algorithm/](https://heycoach.in/blog/applications-of-kruskals-algorithm/)  
28. Kruskal's algorithm \- Wikipedia, accessed August 15, 2025, [https://en.wikipedia.org/wiki/Kruskal%27s\_algorithm](https://en.wikipedia.org/wiki/Kruskal%27s_algorithm)  
29. Kruskal's algorithm (Minimum spanning tree) with real-life examples \- HackerEarth, accessed August 15, 2025, [https://www.hackerearth.com/blog/kruskals-minimum-spanning-tree-algorithm-example](https://www.hackerearth.com/blog/kruskals-minimum-spanning-tree-algorithm-example)  
30. Kruskal's Algorithm and Disjoint Sets \- Csl.mtu.edu, accessed August 15, 2025, [https://www.csl.mtu.edu/cs4321/www/Lectures/Lecture%2019%20-%20Kruskal%20Algorithm%20and%20Dis-joint%20Sets.htm](https://www.csl.mtu.edu/cs4321/www/Lectures/Lecture%2019%20-%20Kruskal%20Algorithm%20and%20Dis-joint%20Sets.htm)