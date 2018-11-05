# Name: Adam Yao      Data: 11/2
import heapq, random, pickle, math, time
from math import pi, acos, sin, cos
import math
from tkinter import *


'''
COLOR KEY:
BLUE IS EXPLORED IN TKINTER PLOT
GREEN IS SHORTEST PATH FROM START TO GOAL

'''


class PriorityQueue():
    """Implementation of a priority queue
    to store nodes during search."""

    # TODO 1 : finish this class

    # HINT look up/use the module heapq.

    def __init__(self):
        self.queue = []
        self.current = 0

    def next(self):
        if self.current >= len(self.queue):
            self.current
            raise StopIteration

        out = self.queue[self.current]
        self.current += 1

        return out

    def pop(self):
        # Your code goes here
        return heapq.heappop(self.queue)

    def remove(self, nodeId):
        # Your code goes here
        return self.remove(nodeId)

    def __iter__(self):
        return self

    def __str__(self):
        return 'PQ:[%s]' % (', '.join([str(i) for i in self.queue]))

    def append(self, node):
        # Your code goes here
        heapq.heappush(self.queue, node)

    def __contains__(self, key):
        self.current = 0
        return key in [n for v, n in self.queue]

    def __eq__(self, other):
        return self == other

    def size(self):
        return len(self.queue)

    def clear(self):
        self.queue = []

    def top(self):
        return self.queue[0]

    __next__ = next


'''Making class Graph(), Node(), and Edge() are optional'''
'''You can make any helper methods'''



def findMin_Max(coordinate):
    ymin = xmin = 1000
    ymax = xmax = -1000
    for key in coordinate:
        y1, x1 = coordinate[key]
        if ymin > float(y1):
            ymin = float(y1)
        if xmin > float(x1):
            xmin = float(x1)
        if ymax < float(y1):
            ymax = float(y1)
        if xmax < float(x1):
            xmax = float(x1)

    return ymin, xmin, ymax, xmax

def make_graph(nodes_file, node_city_file, edge_file):
    '''Make the graph, neighbors, and other dictionaries'''
    graph = {}
    neighbors = {}
    cityToID, IDToCity = {}, {}
    coordinate = {}


#read the lines and manipulate to make convenient dictionary
    for line in open('rrNodeCity.txt', 'r').readlines():
        line = line.strip().split()
        line[1] = " ".join(line[1:])
        cityToID[line[1]] = line[0]
        IDToCity[line[0]] = line[1]
    #coordinate
#setup code to read in lines and create coodinate dicitionary( store coordinates for every node city)
    for line in open('rrNodes.txt', 'r').readlines():
        line = line.strip().split()
        coordinate[line[0]] = line[2], line[1]  ##CHECK THIS


        graph[line[0]] = {}
    for line in open('rrEdges.txt', 'r').readlines(): # make the edges list and finish set up code
        line = line.strip().split()
        id, id2 = line[0], line[1]
        graph[id][id2] = graph[id2][id] = calc_edge_cost(id, id2, coordinate)
        if id not in neighbors:
            neighbors[id] = list([id2])
        else:
            neighbors[id].append(id2)
        if id2 not in neighbors:
            neighbors[id2] = list([id])
        else:
            neighbors[id2].append(id)
    return cityToID, IDToCity, coordinate, graph, neighbors


def calc_edge_cost(start, end, graph):
    # TODO: calculate the edge cost from start city to end city
    #       by using the great circle distance formula.
    #       Refer the distanceDemo.py
    x1, y1 = graph[start]
    x2, y2 = graph[end]
    y1 = float(y1)
    x1 = float(x1)
    y2 = float(y2)
    x2 = float(x2)
    R = 3958.76  # miles = 6371 km
    #
    y1 *= pi / 180.0
    x1 *= pi / 180.0
    y2 *= pi / 180.0
    x2 *= pi / 180.0
    #
    # approximate great circle distance with law of cosines
    #
    value = sin(y1) * sin(y2) + cos(y1) * cos(y2) * cos(x2 - x1)
    '''if value < -1:
        value = -1
    if value > 1:
        value = 1'''
    return acos(value) * R
    #


def breadth_first_search(start, goal, graph):
    # TODO: finish this method
    #       print the number of explored nodes somewhere
    cityToID, IDToCity, coordinate, graphcost, neighbors = graph #create the dictionaries that are comprised of graph
    start = cityToID[start] #make start and goal nodes
    goal = cityToID[goal]
    frontier = [start] #create the frontier
    explored = {start: "s"}
    while len(frontier) > 0: #while frontier length > 0
        current = frontier.pop(0)
        if current == goal:  # if the current node is the goal
            path, cost, length = generate_path(current, explored, coordinate) #create the path, cost, length and return
            return [start] + path, cost, length
        for e in neighbors[current]: #for every neighbor of current node
            if e not in explored:
                frontier.append(e) # append and add to explored
                explored[e] = current
    return None # no solution


def generate_path(n, explored, graph):
    l = [] #create a list
    while explored[n] != "s":
        l.append(n) # append the element to list
        n = explored[n] # add the elemnt to explored
    print()
    l = l[::-1] # reverse the list
    cost = 0
    for i in range(len(l) - 1):
        cost += calc_edge_cost(l[i], l[i + 1], graph) # update the cost
    return l, cost, len(explored) # return the appropriate variables


def dist_heuristic(v, goal, graph):
    # TODO: calculate the heuristic value from node v to the goal
    return calc_edge_cost(v, goal, graph)


def a_starEXT(master, canvas, start, goal, graph, heuristic=dist_heuristic):
    # TODO: Implement A* search algorithm
    #       print the number of explored nodes somewhere

    path_coordinate = {}
    cityToID, IDToCity, coordinate, graphcost, neighbors = graph # create the graph dictionaries
    start = cityToID[start] # start
    goal = cityToID[goal] # goal/stop
    frontier = PriorityQueue()  # make priority queue which will hold all nodes
    frontier.append((heuristic(start, goal, coordinate), start,
                     [start]))  # add first tuple state to frontier with cost 0 at initial start node
    dict = {start: 0}
    if start == goal: return []  # if start node is already equal to goal, return empty set
    explored = set()  # new dictionary with key as node and value as heuristic cost
    # Your code goes here
    while frontier:  # frontier shouldn't be empty unless no Solution, when None is returned
        cost, current, path = frontier.pop()  # seperate tuple into variables
        if current in explored:
            continue
        explored.add(current)
        #DRAW CURRENT

        dict[current] = cost - dist_heuristic(current, goal, coordinate)
        if current == goal:  # if the current node is the goal
            draw_fastest(path, graph, canvas)
            return path, cost, len(explored)  # then you can return the path from the tuple
        for e in neighbors[current]:  # go through every possible child
            if e not in explored:  # if the node hasn't been appended before
                frontier.append((dist_heuristic(e, goal, coordinate) + dict[current] + dist_heuristic(current, e, coordinate), e,path + [e]))  #append the new heuristic and data into frontier
                parent_coordinates = coordinate[e] # parent coordinates for the update_map method
                current_coordinates = coordinate[current] # create the current coordinates
                update_map(parent_coordinates, current_coordinates, graph, canvas, 'blue') # mark all explored lengths


    return None  # no solution


def bidirectional_BFS(start, goal, graph):
    # TODO: finish this method
    #       print the number of explored nodes somewhere
    cityToID, IDToCity, coordinate, graphcost, neighbors = graph # create the appropriate dictionaries
    start = cityToID[start] # start and goal
    goal = cityToID[goal]
    frontier5, frontier2 = [start], [goal] #second frontier
    explored1, explored2 = {start: "s"}, {goal: "s"} # create the explored dictionaries
    counter = 0 # make a counter
    while frontier5 and frontier2:
        if counter % 2 == 0:
            frontier = frontier5 # set frontiers equal
            explored = explored1 # set explored equal
            exploredOpp = explored2 # set explored equal
        else:
            frontier = frontier2 # set frontier 2 equal
            explored = explored2 # set explored equal
            exploredOpp = explored1 # set explored equal
        counter += 1 # update counter
        current = frontier.pop(0) # pop from queue
        if current in exploredOpp:  # if the current node is the goal
            path, cost, length = generate_path(current, explored, coordinate)
            path2, cost2, length2 = generate_path(current, exploredOpp, coordinate)
            return [start] + path + path2[::-1], cost + cost2, length + length2 # return appropriate variables
        for e in neighbors[current]:
            if e not in explored: # if e has not been explored
                frontier.append(e) # append
                explored[e] = current # update
    return None # no solution


def bidirectional_a_star(start, goal, graph, heuristic=dist_heuristic):
    # TODO: Implement bi-directional A*
    #       print the number of explored nodes somewhere
    cityToID, IDToCity, coordinate, graphcost, neighbors = graph
    start = cityToID[start] # start and goal
    goal = cityToID[goal]
    frontier5, frontier2 = PriorityQueue(), PriorityQueue()  # instantiate priority queue
    frontier5.append((heuristic(start, goal, coordinate), start, [start]))
    frontier2.append((heuristic(goal, start, coordinate), goal,
                      [goal]))  # create the first tuple and add it into the frontier
    dict1, dict2 = {start: 0}, {goal: 0} # create the dictionaries
    if start == goal: return []  # return an empty set if start is equal to goal
    explored1 = set()
    explored2 = set()  # new dictionary
    # Your code goes here
    counter = 0
    while frontier5 and frontier2: # while the lengths are greater than 0
        if counter % 2 == 0: # if the counter is even
            frontier = frontier5
            explored = explored1
            dict = dict1
            dictCity = dict2
            point = goal
        else:
            frontier = frontier2
            explored = explored2
            dict = dict2
            point = start
            dictCity = dict1
        cost, current, path = frontier.pop()  # seperate tuple into variables
        if current in explored: #if current in explored
            continue # continue
        explored.add(current) # update explored
        dict[current] = cost - dist_heuristic(current, point, coordinate), path
        if current in dictCity and counter % 2 == 0:  # if the current node is the goal
            return path + dictCity[current][1][::-1], cost, len(explored) # return if the current is in dictCity and counter is even
        elif current in dictCity and counter % 2 == 1:
            return dictCity[current][1] + path[::-1], cost, len(explored)
        counter += 1 # update counter
        for e in neighbors[current]:  # go through every possible child
            if e not in explored:  # if the node hasn't been appended before
                frontier.append((dist_heuristic(e, point, coordinate) + dict[current][0] + dist_heuristic(current, e, coordinate),e, path + [e])) # add into frontier the updated heuristic and nodes
    return None  # no solution

def a_star(start, goal, graph, heuristic=dist_heuristic):
    # TODO: Implement A* search algorithm
    #       print the number of explored nodes somewhere
    cityToID, IDToCity, coordinate, graphcost, neighbors = graph
    start = cityToID[start]
    goal = cityToID[goal]
    frontier = PriorityQueue()  # make priority queue which will hold all nodes
    frontier.append((heuristic(start, goal, coordinate), start,
                     [start]))  # add first tuple state to frontier with cost 0 at initial start node
    dict = {start: 0}
    if start == goal: return []  # return empty set
    explored = set()  # make a new set called explored
    # Your code goes here
    while frontier:  # frontier shouldn't be empty unless no Solution, when None is returned
        cost, current, path = frontier.pop()  # seperate tuple into variables
        if current in explored:
            continue
        explored.add(current)
        dict[current] = cost - dist_heuristic(current, goal, coordinate)
        if current == goal:  # if the current node is the goal
            return path, cost, len(explored)  # return the appropriate variables and data
        for e in neighbors[current]:  # go through every possible child
            if e not in explored:  # if e has not been explored
                frontier.append((dist_heuristic(e, goal, coordinate) + dict[current] + dist_heuristic(current, e,
                                                                                                      coordinate), e,
                                 path + [e]))  # update the frontier with new heuristic and new data
    return None  # no solution


def tridirectional_search(points, graph, heuristic=0):
   # TODO: Do this! Good luck!
   start, middle, end = points # instantiate the start, middle, and end points
   path_startToMiddle, cost1, length1 = a_star(start, middle, graph) # run a star between start and middle
   path_startToEnd, cost2, length2 = a_star(start, end, graph) # run a_star between start and end to check if middle city is worth passing through
   path_endToMiddle, cost3, length3 = a_star(end, middle, graph) # check the path from middle to end and compare
   if (cost1 + cost3) < (cost2 + cost3): # compare the cost of all three cities
       return path_startToMiddle + (path_endToMiddle[::-1])[1:], cost1 + cost3, length1 + length3
   else:
       return path_startToEnd + path_endToMiddle[1:], cost2+ cost3, length2 + length3 # return the appropriate path


def draw_fastest(path, graph, canvas):
    cityToID, IDToCity, coordinate, graphcost, neighbors = graph # get appropriate dictionaries from graph
    counter = 1 # start counter to keep length under the path length
    for x in path:
        if counter < len(path):
            parent_coordinates = coordinate[x]
            current_coordinates = coordinate[path[counter]]
            update_map(parent_coordinates, current_coordinates, graph, canvas, 'green') # update math with the shortest path using the list that contains the shortest path nodes
            counter+=1



def update_map(parent_coordinates, current_coordinates, graph, canvas, color):
    cityToID, IDToCity, coordinate, graphcost, neighbors = graph
    map_width = 1000 # set the map _width
    map_height = 650 # set the map_height
    ymin, xmin, ymax, xmax = findMin_Max(coordinate) # find max and min coordinates for shift
    x_scale, y_scale = map_width / (xmax - xmin + 20.1), map_height / (-ymax - 18.1) # determine y and x scale shift to blow up the map
    xshift = -xmin + 22
    yshift = -ymin  # ymax - float(y2)

    y1, x1 = parent_coordinates # get the parent coordinates
    y2, x2 = current_coordinates # get the current coordinates
    canvas.create_line((float(y1) + yshift) * y_scale, map_width - (float(x1) + xshift) * x_scale,
                       (float(y2) + yshift) * y_scale, map_width - (float(x2) + xshift) * x_scale, fill=color) # draw the line
    canvas.update()

def draw_map(graph, master, canvas):
    cityToID, IDToCity, coordinate, graphcost, neighbors = graph
    map_width = 1000  # set the map _width
    map_height = 650  # set the map_height
    ymin, xmin, ymax, xmax = findMin_Max(coordinate)  # find max and min coordinates for shift
    x_scale, y_scale = map_width / (xmax - xmin + 20.1), map_height / (
                -ymax - 18.1)  # determine y and x scale shift to blow up the map
    xshift = -xmin + 22
    yshift = -ymin  # ymax - float(y2)

    for key1 in neighbors:
        key_coordinates = coordinate[key1]
        y1, x1 = key_coordinates # get the coordinates of USA
        children = neighbors[key1]
        for val in children:
            val_coordinates = coordinate[val]
            y2, x2 = val_coordinates

            canvas.create_line( (float(y1) + yshift) * y_scale, map_width - (float(x1) + xshift) * x_scale,
                           (float(y2) + yshift) * y_scale, map_width -  (float(x2) + xshift) * x_scale, fill='red') # draw US in white


    canvas.pack() # pack and update
    canvas.update()



def main():
    start = input("Start city: ")
    goal = input("Goal city: ")

    '''depends on your data setup, you can change this part'''
    graph = make_graph("rrNodes.txt", "rrNodeCity.txt", "rrEdges.txt")
    coordinate = graph[1]




    print("\nBFS Summary")
    cur_time = time.time()
    bfs_path, cost, len = breadth_first_search(start, goal, graph)
    next_time = time.time()
    print("Cost: ", cost)
    print("BFS path: ", bfs_path)
    print("Length of explored: ", len)
    l = []
    for e in bfs_path:
        if e in graph[1]:
            l.append(graph[1][e])
    print("City Path: ", l)
    print("BFS Duration: ", (next_time - cur_time))

    print("\nA* Search Summary")
    cur_time = time.time()
    a_star_path, cost, len = a_star(start, goal, graph)


    next_time = time.time()
    print("Cost: ", cost)
    print("A* path: ", a_star_path)
    print("Length of explored: ", len)
    l = []
    for e in a_star_path:
        if e in graph[1]:
            l.append(graph[1][e])
    print("City Path: ", l)
    print("A* Duration: ", (next_time - cur_time))

    print("\nBi-directional BFS Summary")
    cur_time = time.time()
    bi_path, cost, len = bidirectional_BFS(start, goal, graph)
    next_time = time.time()
    print("Cost: ", cost)
    print("Bi-directional BFS path: ", a_star_path)
    print("Length of explored: ", len)
    l = []
    for e in a_star_path:
        if e in graph[1]:
            l.append(graph[1][e])
    print("City Path: ", l)
    print("Bi-directional BFS Duration: ", (next_time - cur_time))

    print("\nBi-directional A* Summary")
    cur_time = time.time()
    bi_a_path, cost, len = bidirectional_a_star(start, goal, graph)
    next_time = time.time()
    print("Cost: ", cost)
    print("Bi-directional A* path: ", bi_a_path)
    print("Length of explored: ", len)
    l = []
    for e in a_star_path:
        if e in graph[1]:
            l.append(graph[1][e])
    print("City Path: ", l)
    print("Bi-directional A* Duration: ", (next_time - cur_time))

    master = Tk()
    w = Canvas(master, width=1110, height=690)
    master.resizable(False, False)

    master.title("United States Railroad Map")
    w.configure(background='white')
    w.pack()
    draw_map(graph, master, w)
    a_starEXT(master, w, start, goal, graph) #master, canvas, start, goal, graph, heuristic=dist_heuristic):
    master.mainloop()

    # TODO: check your tridirectional search algorithm here



if __name__ == '__main__':
    main()

'''Sample Run
Start city: Los Angeles
Goal city: Chicago

BFS Summary
cost:  2093.868463307088
node path:  ['0600316', '0600089', '0600426', '0600087', '0600531', '0600760', '0600411', '0600027', '0600590', '0600023', '0600899', '0600900', '0600901', '0600902', '0600035', '0600321', '0600769', '0600436', '0600032', '0600414', '0600867', '0600866', '0600031', '0600033', '0600795', '0600602', '0600603', '0600036', '0600604', '0600871', '0600870', '0600872', '0600495', '0000144', '0400113', '0400114', '0400009', '0400010', '0400116', '0400117', '0400148', '0400074', '0400146', '0400147', '0400064', '0400005', '0400006', '0400063', '0400100', '0400075', '0400071', '0400070', '0400002', '0400050', '0000312', '3500036', '3500062', '3500063', '3500068', '3500069', '3500101', '3500111', '3500061', '3500109', '3500084', '3500089', '3500102', '3500065', '3500066', '3500032', '3500027', '3500119', '3500071', '3500070', '3500090', '3500107', '3500072', '3500013', '3500047', '3500039', '3500141', '3500025', '3500099', '0000257', '4801203', '4800003', '4801200', '4800002', '0000248', '4000264', '4000138', '4000231', '0000246', '2000206', '2000503', '2000360', '2000427', '2000500', '2000452', '2000207', '2000419', '2000501', '2000502', '2000073', '2000074', '2000075', '2000473', '2000519', '2000505', '2000291', '2000289', '2000290', '2000288', '2000292', '2000298', '2000087', '2000093', '2000094', '2000095', '2000096', '2000135', '2000280', '2000133', '2000342', '2000439', '2000358', '2000134', '2000121', '2000442', '2000441', '2000124', '2000125', '2000271', '2000127', '2000272', '2000237', '2000273', '2000353', '2000220', '0000541', '2900116', '2900283', '2900235', '2900198', '2900286', '2900241', '2900103', '2900482', '2900102', '2900545', '2900556', '2900111', '2900120', '2900122', '2900494', '2900355', '2900121', '2900162', '2900165', '2900566', '2900468', '2900164', '0000395', '1900057', '1900382', '1900070', '0000393', '1701225', '1700286', '1701010', '1701170', '1700285', '1701321', '1701322', '1700287', '1700296', '1701472', '1700303', '1700328', '1700926', '1700582', '1700310', '1700311', '1700312', '1700583', '1700313', '1701182', '1701345', '1700327', '1700432', '1701622', '1700449', '1700419', '1700465', '1700418', '1701034', '1701194', '1700417', '1700629', '1701394', '1700653', '1700631', '1700415', '1701267', '1701265', '1701291']
number of explored:  13268
BFS path:  ['Los Angeles', 'Chicago']
BFS Duration:  0.03057575225830078

A* Search Summary
cost:  2002.0784404122933
node path:  ['0600316', '0600427', '0600322', '0600751', '0600084', '0600685', '0600085', '0600080', '0600079', '0600686', '0600766', '0600402', '0600799', '0600408', '0600460', '0600588', '0600384', '0600688', '0600463', '0600435', '0600107', '0600775', '0600769', '0600436', '0600032', '0600414', '0600867', '0600866', '0600031', '0600033', '0600795', '0600602', '0600603', '0600036', '0600604', '0600871', '0600870', '0600872', '0600495', '0000144', '0400113', '0400114', '0400009', '0400010', '0400116', '0400117', '0400148', '0400074', '0400146', '0400147', '0400064', '0400005', '0400006', '0400063', '0400100', '0400075', '0400071', '0400070', '0400002', '0400050', '0000312', '3500036', '3500062', '3500063', '3500068', '3500069', '3500101', '3500111', '3500061', '3500109', '3500084', '3500089', '3500102', '3500065', '3500066', '3500032', '3500027', '3500119', '3500071', '3500070', '3500090', '3500107', '3500072', '3500013', '3500047', '3500039', '3500141', '3500025', '3500099', '0000257', '4801203', '4800003', '4801200', '4800002', '0000248', '4000264', '4000138', '4000231', '0000246', '2000206', '2000503', '2000360', '2000427', '2000500', '2000452', '2000207', '2000419', '2000501', '2000502', '2000073', '2000074', '2000075', '2000473', '2000519', '2000506', '2000294', '2000295', '2000296', '2000514', '2000523', '2000077', '2000292', '2000504', '2000293', '2000092', '2000311', '2000472', '2000470', '2000094', '2000095', '2000404', '2000097', '2000277', '2000102', '2000414', '2000103', '2000104', '2000106', '2000356', '2000114', '2000372', '2000117', '2000465', '2000466', '2000467', '2000270', '2000258', '2000257', '2000256', '2000260', '0000232', '2900371', '2900374', '2900378', '2900238', '2900184', '2900358', '2900343', '2900206', '2900095', '2900598', '2900476', '2900101', '2900212', '2900100', '2900106', '2900281', '2900210', '2900290', '2900291', '2900292', '2900207', '2900558', '2900416', '2900493', '2900253', '2900121', '2900162', '2900165', '2900566', '2900468', '2900164', '0000395', '1900057', '1900382', '1900070', '0000393', '1701225', '1700286', '1701010', '1701170', '1700285', '1701321', '1701325', '1701326', '1701323', '1700750', '1701328', '1701327', '1700292', '1700281', '1700280', '1701120', '1700301', '1700922', '1701121', '1700487', '1700480', '1700479', '1700478', '1700477', '1700430', '1700431', '1701157', '1700449', '1700419', '1700465', '1700418', '1701034', '1701194', '1700417', '1700629', '1701394', '1700653', '1700631', '1700415', '1701267', '1701265', '1701291']
number of explored:  1272
A* path:  ['Los Angeles', 'Chicago']
A* Duration:  0.036072492599487305

Bi-directional BFS Summary
cost:  2093.868463307088
node path:  ['0600316', '0600089', '0600426', '0600087', '0600531', '0600760', '0600411', '0600027', '0600590', '0600023', '0600899', '0600900', '0600901', '0600902', '0600035', '0600321', '0600769', '0600436', '0600032', '0600414', '0600867', '0600866', '0600031', '0600033', '0600795', '0600602', '0600603', '0600036', '0600604', '0600871', '0600870', '0600872', '0600495', '0000144', '0400113', '0400114', '0400009', '0400010', '0400116', '0400117', '0400148', '0400074', '0400146', '0400147', '0400064', '0400005', '0400006', '0400063', '0400100', '0400075', '0400071', '0400070', '0400002', '0400050', '0000312', '3500036', '3500062', '3500063', '3500068', '3500069', '3500101', '3500111', '3500061', '3500109', '3500084', '3500089', '3500102', '3500065', '3500066', '3500032', '3500027', '3500119', '3500071', '3500070', '3500090', '3500107', '3500072', '3500013', '3500047', '3500039', '3500141', '3500025', '3500099', '0000257', '4801203', '4800003', '4801200', '4800002', '0000248', '4000264', '4000138', '4000231', '0000246', '2000206', '2000503', '2000360', '2000427', '2000500', '2000452', '2000207', '2000419', '2000501', '2000502', '2000073', '2000074', '2000075', '2000473', '2000519', '2000505', '2000291', '2000289', '2000290', '2000288', '2000292', '2000298', '2000087', '2000093', '2000094', '2000095', '2000096', '2000135', '2000280', '2000133', '2000342', '2000439', '2000358', '2000134', '2000121', '2000442', '2000441', '2000124', '2000125', '2000271', '2000127', '2000272', '2000237', '2000273', '2000353', '2000220', '0000541', '2900116', '2900283', '2900235', '2900198', '2900286', '2900241', '2900103', '2900482', '2900102', '2900545', '2900556', '2900111', '2900120', '2900122', '2900494', '2900355', '2900121', '2900162', '2900165', '2900566', '2900468', '2900164', '0000395', '1900057', '1900382', '1900070', '0000393', '1701225', '1700286', '1701010', '1701170', '1700285', '1701321', '1701322', '1700287', '1700296', '1701472', '1700303', '1700328', '1700926', '1700582', '1700310', '1700311', '1700312', '1700583', '1700313', '1701182', '1701345', '1700327', '1700432', '1701622', '1700449', '1700419', '1700465', '1700418', '1701034', '1701194', '1700417', '1700629', '1701394', '1700653', '1700631', '1700415', '1701267', '1701265', '1701291']
number of explored:  ### Not Shown ###
Bi-directional BFS path:  ['Los Angeles', 'Chicago']
Bi-directional BFS Duration:  0.0

Bi-directional A* Summary
cost:  2002.0784404122933
node path:  ['0600316', '0600427', '0600322', '0600751', '0600084', '0600685', '0600085', '0600080', '0600079', '0600686', '0600766', '0600402', '0600799', '0600408', '0600460', '0600588', '0600384', '0600688', '0600463', '0600435', '0600107', '0600775', '0600769', '0600436', '0600032', '0600414', '0600867', '0600866', '0600031', '0600033', '0600795', '0600602', '0600603', '0600036', '0600604', '0600871', '0600870', '0600872', '0600495', '0000144', '0400113', '0400114', '0400009', '0400010', '0400116', '0400117', '0400148', '0400074', '0400146', '0400147', '0400064', '0400005', '0400006', '0400063', '0400100', '0400075', '0400071', '0400070', '0400002', '0400050', '0000312', '3500036', '3500062', '3500063', '3500068', '3500069', '3500101', '3500111', '3500061', '3500109', '3500084', '3500089', '3500102', '3500065', '3500066', '3500032', '3500027', '3500119', '3500071', '3500070', '3500090', '3500107', '3500072', '3500013', '3500047', '3500039', '3500141', '3500025', '3500099', '0000257', '4801203', '4800003', '4801200', '4800002', '0000248', '4000264', '4000138', '4000231', '0000246', '2000206', '2000503', '2000360', '2000427', '2000500', '2000452', '2000207', '2000419', '2000501', '2000502', '2000073', '2000074', '2000075', '2000473', '2000519', '2000506', '2000294', '2000295', '2000296', '2000514', '2000523', '2000077', '2000292', '2000504', '2000293', '2000092', '2000311', '2000472', '2000470', '2000094', '2000095', '2000404', '2000097', '2000277', '2000102', '2000414', '2000103', '2000104', '2000106', '2000356', '2000114', '2000372', '2000117', '2000465', '2000466', '2000467', '2000270', '2000258', '2000257', '2000256', '2000260', '0000232', '2900371', '2900374', '2900378', '2900238', '2900184', '2900358', '2900343', '2900206', '2900095', '2900598', '2900476', '2900101', '2900212', '2900100', '2900106', '2900281', '2900210', '2900290', '2900291', '2900292', '2900207', '2900558', '2900416', '2900493', '2900253', '2900121', '2900162', '2900165', '2900566', '2900468', '2900164', '0000395', '1900057', '1900382', '1900070', '0000393', '1701225', '1700286', '1701010', '1701170', '1700285', '1701321', '1701325', '1701326', '1701323', '1700750', '1701328', '1701327', '1700292', '1700281', '1700280', '1701120', '1700301', '1700922', '1701121', '1700487', '1700480', '1700479', '1700478', '1700477', '1700430', '1700431', '1701157', '1700449', '1700419', '1700465', '1700418', '1701034', '1701194', '1700417', '1700629', '1701394', '1700653', '1700631', '1700415', '1701267', '1701265', '1701291']
number of explored:  ### Not Shown ###
Bi-directional A* path:  ['Los Angeles', 'Tucson', 'Fort Worth', 'Chicago']
Bi-directional A* Duration:  0.0
'''
