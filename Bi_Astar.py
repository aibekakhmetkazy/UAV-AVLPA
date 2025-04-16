import math
import heapq


def pathplanningBidirectionalAStar(adjacency_list, coordinates, N, start=None, goal=None):
    """
    Discrete Bidirectional A* path planning on a graph.

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

    Returns:
      (full_path, total_distance): tuple
          full_path: A list of vertex indices from start to goal if a path is found; otherwise, returns None.
          total_distance: The total Euclidean distance along the path (None if no path is found).
    """

    # Set default start and goal if not provided.
    if start is None:
        start = N - 1
    if goal is None:
        goal = N

    def euclidean_distance(i, j):
        """Compute the Euclidean distance between vertices i and j using their coordinates."""
        (x1, y1) = coordinates[i]
        (x2, y2) = coordinates[j]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # Heuristic functions:
    # For forward search, h(n) = distance from n to goal.
    # For backward search, h(n) = distance from n to start.
    h_forward = lambda n: euclidean_distance(n, goal)
    h_backward = lambda n: euclidean_distance(n, start)

    # Priority queues for the two searches: each element is a tuple (f, node).
    open_forward = []
    open_backward = []

    # g-scores for the two searches.
    g_forward = {start: 0}
    g_backward = {goal: 0}

    # Parent dictionaries for path reconstruction.
    came_from_forward = {start: None}
    came_from_backward = {goal: None}

    # Push the start and goal nodes into their respective open lists.
    heapq.heappush(open_forward, (g_forward[start] + h_forward(start), start))
    heapq.heappush(open_backward, (g_backward[goal] + h_backward(goal), goal))

    # Closed sets to track visited nodes.
    closed_forward = set()
    closed_backward = set()

    best_cost = float('inf')
    meeting_node = None

    # Main loop: run until one of the open lists is empty.
    while open_forward and open_backward:
        # Check termination condition:
        # If the sum of the minimum f-values from both searches is no less than the best found cost, stop.
        min_forward = open_forward[0][0]
        min_backward = open_backward[0][0]
        if min_forward + min_backward >= best_cost:
            break

        # --- Expand forward search ---
        if open_forward:
            f_val, current = heapq.heappop(open_forward)
            if current in closed_forward:
                continue
            closed_forward.add(current)

            # Check if the current node was already expanded in the backward search.
            if current in closed_backward:
                total = g_forward[current] + g_backward[current]
                if total < best_cost:
                    best_cost = total
                    meeting_node = current

            # Expand neighbors in the forward direction.
            for neighbor_str, weight in adjacency_list.get(current, []):
                if neighbor_str > N:
                    continue
                neighbor = int(neighbor_str)
                tentative_g = g_forward[current] + weight
                if neighbor not in g_forward or tentative_g < g_forward[neighbor]:
                    g_forward[neighbor] = tentative_g
                    came_from_forward[neighbor] = current
                    f_neighbor = tentative_g + h_forward(neighbor)
                    heapq.heappush(open_forward, (f_neighbor, neighbor))
                    # If the neighbor has already been processed in the backward search, update best_cost.
                    if neighbor in closed_backward:
                        total = g_forward[neighbor] + g_backward[neighbor]
                        if total < best_cost:
                            best_cost = total
                            meeting_node = neighbor

        # --- Expand backward search ---
        if open_backward:
            f_val, current = heapq.heappop(open_backward)
            if current in closed_backward:
                continue
            closed_backward.add(current)

            if current in closed_forward:
                total = g_forward[current] + g_backward[current]
                if total < best_cost:
                    best_cost = total
                    meeting_node = current

            # Expand neighbors in the backward direction.
            for neighbor_str, weight in adjacency_list.get(current, []):
                if neighbor_str > N:
                    continue
                neighbor = int(neighbor_str)
                tentative_g = g_backward[current] + weight
                if neighbor not in g_backward or tentative_g < g_backward[neighbor]:
                    g_backward[neighbor] = tentative_g
                    came_from_backward[neighbor] = current
                    f_neighbor = tentative_g + h_backward(neighbor)
                    heapq.heappush(open_backward, (f_neighbor, neighbor))
                    if neighbor in closed_forward:
                        total = g_forward[neighbor] + g_backward[neighbor]
                        if total < best_cost:
                            best_cost = total
                            meeting_node = neighbor

    # If no meeting node was found, there is no path.
    if meeting_node is None:
        print("No path found.")
        return None, None

    # --- Reconstruct the path ---
    # Reconstruct the path from start to meeting_node.
    path_forward = []
    node = meeting_node
    while node is not None:
        path_forward.append(node)
        node = came_from_forward.get(node)
    path_forward.reverse()  # Now from start to meeting_node.

    # Reconstruct the path from meeting_node to goal.
    path_backward = []
    node = came_from_backward.get(meeting_node)
    while node is not None:
        path_backward.append(node)
        node = came_from_backward.get(node)

    # Concatenate the forward and backward paths (avoid duplicating the meeting node).
    full_path = path_forward + path_backward

    # Compute the total Euclidean distance along the path.
    total_distance = 0.0
    for i in range(len(full_path) - 1):
        total_distance += euclidean_distance(full_path[i], full_path[i + 1])

    # print("Bidirectional A* found a path:\n", full_path, end='\n')
    # print("Total distance:", round(total_distance, 3), end='\n')
    # print('------------------------------------------------')
    return full_path