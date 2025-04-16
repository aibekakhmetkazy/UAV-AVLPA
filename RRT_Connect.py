import math
import random


def pathplanningRRT(adjacency_list, coordinates, N, start=None, goal=None, max_iter=5000):
    """
    Discrete RRT-Connect path planning on a graph.

    Parameters:
      adjacency_list: dict
          Keys are vertex identifiers as strings (e.g., '1') and values are lists of tuples (neighbor, weight).
      coordinates: list
          List of [x, y] coordinates, with coordinates[0] unused (i.e., vertices indexed 1..N).
      N: int
          Total number of vertices (len(coordinates)-1).
      start: int, optional
          Starting vertex index. Defaults to N-1.
      goal: int, optional
          Goal vertex index. Defaults to N.
      max_iter: int, optional
          Maximum number of iterations to try before giving up.

    Returns:
      full_path: list
          A list of vertex indices from start to goal if a path is found; otherwise, returns None.
    """

    # Set default start and goal if not provided.
    if start is None:
        start = N - 1
    if goal is None:
        goal = N

    def euclidean_distance(i, j):
        """Compute Euclidean distance between vertices i and j."""
        (x1, y1) = coordinates[i]
        (x2, y2) = coordinates[j]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def nearest(tree, q):
        """
        Given a tree (a dict of vertices with parent pointers) and a target vertex q,
        return the vertex in the tree that is closest to q in Euclidean distance.
        """
        best = None
        best_dist = float('inf')
        for v in tree.keys():
            d = euclidean_distance(v, q)
            if d < best_dist:
                best_dist = d
                best = v
        return best

    def steer(v, q):
        """
        From vertex v, choose a neighbor that is closer to target q (if one exists).
        Returns the neighbor (vertex index) that minimizes the distance to q,
        provided it is closer than v itself. Otherwise, returns None.
        """
        current_dist = euclidean_distance(v, q)
        best_neighbor = None
        best_dist = current_dist
        # The neighbors are stored in the adjacency list with keys as strings.
        for neighbor_str, weight in adjacency_list[v]:
            if neighbor_str > N:
                break
            neighbor = int(neighbor_str)
            d = euclidean_distance(neighbor, q)
            if d < best_dist:
                best_neighbor = neighbor
                best_dist = d
        return best_neighbor

    # Initialize two trees: one rooted at the start and one at the goal.
    T_start = {start: None}  # key: vertex, value: parent (None for the root)
    T_goal = {goal: None}

    for iteration in range(max_iter):
        # ---- Extend T_start ----
        q_rand = random.randint(1, N)
        q_near = nearest(T_start, q_rand)
        q_new = steer(q_near, q_rand)
        if q_new is not None and q_new not in T_start:
            T_start[q_new] = q_near
            # Now attempt to connect T_goal toward q_new
            q_connect = q_new
            while True:
                q_near_goal = nearest(T_goal, q_connect)
                q_new_goal = steer(q_near_goal, q_connect)
                # If no further progress can be made or already visited, break.
                if q_new_goal is None or q_new_goal in T_goal:
                    break
                T_goal[q_new_goal] = q_near_goal
                q_connect = q_new_goal
                # If the two trees meet (or are nearly equal), we found a connection.
                if q_connect == q_new or euclidean_distance(q_connect, q_new) < 1e-6:
                    # Reconstruct path from start to q_new in T_start.
                    path_from_start = []
                    v = q_new
                    while v is not None:
                        path_from_start.append(v)
                        v = T_start[v]
                    path_from_start.reverse()  # now goes from start -> q_new

                    # Reconstruct path from connection to goal in T_goal.
                    path_from_goal = []
                    v = q_connect
                    while v is not None:
                        path_from_goal.append(v)
                        v = T_goal[v]
                    # Combine paths (avoid duplicating the connection vertex)
                    full_path = path_from_start + path_from_goal[1:]

                    total_distance = 0.0
                    for i in range(len(full_path) - 1):
                        total_distance += euclidean_distance(full_path[i], full_path[i+1])

                    # print("d-RRT Connect found a path:\n", full_path, end="\n")
                    # print('Total distance:', round(total_distance, 3))
                    # print('------------------------------------------------')

                    return full_path
        # ---- Swap the trees to alternate extension ----
        T_start, T_goal = T_goal, T_start

    # If no connection was found within max_iter, return None.
    return None
