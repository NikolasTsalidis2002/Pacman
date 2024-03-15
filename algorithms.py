import sys

def dijkstra(nodes, start_node):
    unvisited_nodes = list(nodes.costs)
    shortest_path = {}
    previous_nodes = {}
    # add the highest pontetial cost of travelling thorugh all nodes in the system
    # make the start node's value = 0
    max_value = sys.maxsize
    for node in unvisited_nodes:
        shortest_path[node] = max_value
    shortest_path[start_node] = 0
    # do this until we have visited the whole network
    while unvisited_nodes:
        current_min_node = None
        for node in unvisited_nodes:
            if current_min_node == None:
                current_min_node = node
            elif shortest_path[node] < shortest_path[current_min_node]:
                current_min_node = node
        # go through all the neighbors of the node with the 
        neighbors = nodes.getNeighbors(current_min_node)
        for neighbor in neighbors:
            tentative_value = shortest_path[current_min_node] + 1 #nodes.value(current_min_node, neighbor)
            if tentative_value < shortest_path[neighbor]:  
                shortest_path[neighbor] = tentative_value
                # We also update the best path to the current node
                previous_nodes[neighbor] = current_min_node
 
        # After visiting its neighbors, we mark the node as "visited"
        unvisited_nodes.remove(current_min_node)
    
    return previous_nodes, shortest_path



#########
# A*
def heuristic(node1, node2):
    # manhattan distance
    return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])


def dijkstra_or_a_star(nodes, start_node, a_star=False):
    unvisited_nodes = list(nodes.costs)
    shortest_path = {}
    previous_nodes = {}

    max_value = sys.maxsize
    for node in unvisited_nodes:
        shortest_path[node] = max_value
    shortest_path[start_node] = 0

    while unvisited_nodes:
        current_min_node = None
        for node in unvisited_nodes:
            if current_min_node == None:
                current_min_node = node
            elif shortest_path[node] < shortest_path[current_min_node]:
                current_min_node = node

        neighbors = nodes.getNeighbors(current_min_node)
        for neighbor in neighbors:
            if a_star:
                tentative_value = shortest_path[current_min_node] + heuristic(current_min_node,neighbor) 
            else:
                tentative_value = shortest_path[current_min_node] + 1
            try:
                if tentative_value < shortest_path[neighbor]:
                    shortest_path[neighbor] = tentative_value
                    # We also update the best path to the current node
                    previous_nodes[neighbor] = current_min_node
            except Exception as e:
                # print('WARNING! There is a node that could not be found from the list of neighbors',e)
                pass
        # After visiting its neighbors, we mark the node as "visited"
        unvisited_nodes.remove(current_min_node)    
    return previous_nodes, shortest_path