from pyroutelib3 import Router, Datastore # Import the router
import folium
import xmltodict
import os 
import html
import networkx as nx
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
import random
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import draw_hwn_map
from copy import deepcopy

hwn_file_name = os.path.join("hwn_gpx", "HWN_2020_05_01.gpx")
done_stamps_folder = "stamps"
perfect_tour_length = 15.0

def get_visited_dict():
    files = os.listdir(done_stamps_folder)
    all_stamps_lists = {}
    for f in files:
        if not f.endswith(".txt"):
            continue
        stamps_list = []
        with open(os.path.join(done_stamps_folder, f)) as fd:
            for line in fd:
                stamps_list.append("HWN"+line.strip())
        all_stamps_lists[f.replace(".txt", "")] = (stamps_list, "red")
    return all_stamps_lists


# Taken from https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
def haversine(lon1, lat1, lon2, lat2):
       """
       Calculate the great circle distance between two points 
       on the earth (specified in decimal degrees)
       """
       # convert decimal degrees to radians 
       lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
       # haversine formula 
       dlon = lon2 - lon1 
       dlat = lat2 - lat1 
       a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
       c = 2 * asin(sqrt(a)) 
       # Radius of earth in kilometers is 6371
       km = 6371* c
       return km

def eval_tour(G, tour):
    if len(tour) <= 1:
        return 20.0 # magic!
    tour_length = 0.0
    for i in range(len(tour)):
        from_node = tour[i]
        to_node = tour[(i+1) % len(tour)]
        if from_node == to_node:
            print("WARNING: Inconsistent tour:", tour)
        if from_node < to_node:
            tour_length += G.edges[from_node, to_node]['weight']
        else:
            tour_length += G.edges[to_node, from_node]['weight']
    return tour_length

def create_random_tour(G):
    tour_nodes = 3
    limit = len(G.nodes)
    tour = []
    for i in range(tour_nodes):
        while True:
            chosen = random.randint(1, limit)
            if not chosen in tour:
                tour.append(chosen) 
                break
    return tour

def create_population(G):
    to_be_chosen = set()
    for i in range(0, len(G.nodes)):
        if G.nodes[i+1]['visitors_cnt']==0:
            to_be_chosen.add(i+1)

    population = []
    while(len(to_be_chosen) > 0):
        tour_length = random.randint(1, 5)
        new_tour = []
        for i in range(tour_length):
            next_choice = random.choice(tuple(to_be_chosen))
            to_be_chosen.remove(next_choice)
            new_tour.append(next_choice)
            if len(to_be_chosen) == 0:
                break
        population.append(new_tour)
    # print("population=", population)
    return population
        

def prepare_graph(filter=None, limit = None):
    with open(hwn_file_name, encoding="utf-8") as fd:
        doc = xmltodict.parse(fd.read())
        
        G = nx.Graph()
         
        bounds = doc["gpx"]["metadata"]["bounds"]  
        center_lat = (float(bounds["@maxlat"]) + float(bounds["@minlat"])) / 2.0
        center_lon = (float(bounds["@maxlon"]) + float(bounds["@minlon"])) / 2.0
        
        m = folium.Map(location=[center_lat, center_lon])
        
        stamps = doc["gpx"]["wpt"]
        for s in stamps:
            caption = s["name"]
            visitors_cnt = 0
            if filter:
                for k, v in filter.items():
                    if s["name"] in v[0]:
                        visitors_cnt += 1
 
            G.add_node(int(caption.replace("HWN","")), lat=float(s["@lat"]), lon=float(s["@lon"]), visitors_cnt = visitors_cnt)
            if limit and int(caption.replace("HWN","")) == limit:
                break

    count = 0
    for i in range(len(G.nodes)):
        for j in range(i, len(G.nodes)):
            if i==j:
                continue
            dist = haversine(G.nodes[i+1]['lon'], G.nodes[i+1]['lat'], G.nodes[j+1]['lon'], G.nodes[j+1]['lat'])
            G.add_edge(i+1, j+1, weight=dist )
    return G

def draw_graph(G):
    pos = nx.spring_layout(G, weight="weight")
    nx.draw(G, with_labels=True, pos=pos)
    plt.show()

def count_visits(G, tour):
    tour_visits = 0
    for i in range(len(tour)):
        tour_visits += G.nodes[i+1]['visitors_cnt']
    return tour_visits
            
            
def evaluate_population(G, population):
    """
    perfect_length = 15.0 # direct distances!
    total_len_km = 0.0
    total_derivation_km = 0.0
    total_len_tours = len(population)
    for i, t in enumerate(population):
        tour_length = eval_tour(G, t)
        #print("[{}] tour=".format(i),t)
        #print("[{}] tour_length=".format(i), tour_length)
        total_len_km += tour_length

        total_derivation_km += 2**abs(perfect_length - tour_length) 
        #if abs(perfect_length - tour_length) > 5.0:
        #    total_derivation_km += 1000
        tour_visits = count_visits(G, t)

    fitness = total_len_km 
    #if not is_valid_solution(G, population): # # + total_derivation_km # total_derivation_km*total_len_tours*(tour_visits+1)
    #    fitness += 1000
    #fitness = total_derivation_km + 10*total_len_tours + tour_visits
    """
    sum_fitness = 0.0
    for i, t in enumerate(population):
        sum_fitness += route_fitness(G, t)
    
    return sum_fitness #/ len(population)
  

def sort_valid_solution(G, route):
    # perfect_tour_length = 15.0
    length = eval_tour(G, route)
    if length < perfect_tour_length*0.5 or length > perfect_tour_length*1.5:
        return 1
    return 0
    
def route_fitness(G, route):
    #perfect_length = 15.0 # direct distances!
    # perfect_stops = 4
    # return (abs(perfect_tour_length - eval_tour(G, route)) + 1)*(abs(perfect_stops - len(route)) + 1)
    return eval_tour(G, route) / len(route) + (abs(perfect_tour_length - eval_tour(G, route)))
    
def crossover(G, population):
    p = sorted(population, key=lambda x: sort_valid_solution(G, x))
    # p = sorted(population, key=lambda x: route_fitness(G, x))
    
    crossover_index = random.randint(0, len(population)-1)
    for i, t in enumerate(p):
        is_valid = sort_valid_solution(G, t)
        #print("  ", t , " valid=", is_valid, " len=", eval_tour(G, t))
        if is_valid == 1 and i>0:
            crossover_index = i-1 # 
            break
    #print("sorted=",p )
    
    
    # now p is sorted
    # crossover_index = random.randint(0, len(population)-1)
    
    #print("crossover at idx:", crossover_index)
    p_new = p[0:crossover_index]
    p_reorder = p[crossover_index:]
    
    to_be_chosen = []
    for p in p_reorder:
        to_be_chosen.extend(p)

    while len(to_be_chosen) > 0:
        random_length = random.randint(1, min(5, len(to_be_chosen)))
        new_tour = []
        for i in range(random_length):
            random_idx = random.randint(0, len(to_be_chosen)-1)
            new_tour.append(to_be_chosen[random_idx])
            del to_be_chosen[random_idx]
        """new_tour = to_be_chosen[:random_length]
        print("new_tour=", new_tour)
        for i in range(random_length):
            del to_be_chosen[0]
        """
        p_new.append(new_tour)
    return p_new
  
def evolution_step(old_population):
    population = deepcopy(old_population)
    r = random.randint(0, 7) # both inclusive
    if r == 0:
        # arbitrary positions changing
        affected_tour_index = random.randint(0, len(population)-1)
        from_index = random.randint(0, len(population[affected_tour_index])-1)
        to_index = random.randint(0, len(population[affected_tour_index])-1)
        population[affected_tour_index][from_index], population[affected_tour_index][to_index] = population[affected_tour_index][to_index], population[affected_tour_index][from_index]

    elif r == 1:
        # neighboring position change
        affected_tour_index = random.randint(0, len(population)-1)
        affected_index = random.randint(0, len(population[affected_tour_index])-1)
        other_index = (affected_index+1) % len(population[affected_tour_index])
        population[affected_tour_index][affected_index], population[affected_tour_index][other_index] = population[affected_tour_index][other_index], population[affected_tour_index][affected_index]
    elif 2 <= r <= 5:
        # switch between tours
        tour_index_1 = random.randint(0, len(population)-1)
        tour_index_2 = random.randint(0, len(population)-1)
        if tour_index_1 == tour_index_2:
            return population
        element_index_1 = random.randint(0, len(population[tour_index_1])-1)
        element_index_2 = random.randint(0, len(population[tour_index_2])-1)
        if element_index_1 == element_index_2:
            return population
        population[tour_index_1][element_index_1], population[tour_index_2][element_index_2] = population[tour_index_2][element_index_2], population[tour_index_1][element_index_1]
    elif 6 <= r <= 6:
        # split tour
        tour_index = random.randint(0, len(population)-1)
        if len(population[tour_index]) > 1:
            element_index = random.randint(1, len(population[tour_index])-1)
            #print("split: ", population[tour_index])
            #print("-->    ", population[tour_index][:element_index], population[tour_index][element_index:])
            population.append(population[tour_index][element_index:])
            population[tour_index] = population[tour_index][:element_index]
    else:
        # merge tours
        if len(population) >= 2:
            tour_index_1 = random.randint(0, len(population)-1)
            tour_index_2 = random.randint(0, len(population)-1)
            if tour_index_1 != tour_index_2:
                population[tour_index_1].extend(population[tour_index_2])
                del population[tour_index_2]
        
    return population
    
def pretty_print_population(G, population):
    print("fitness_after = ", evaluate_population(G, population))
    for t in population:
        print(" length=", eval_tour(G, t), "t=",t)
    
def get_routes(population):
    route_list = []
    for t in population:
        route_list.append(t)
    return route_list

def is_valid_solution(G, population):
    # perfect_tour_length = 15.0
    imperfect_tours = 0
    for t in population:
        length = eval_tour(G, t)
        if length < perfect_tour_length*0.5 or length > perfect_tour_length*1.5:
            imperfect_tours += 1
    return imperfect_tours
    

G = prepare_graph(filter=get_visited_dict(), limit=222)
#tour = create_random_tour(G)
#tour_length = eval_tour(G, tour)
#print("tour=",tour)
#print("tour_length=",tour_length)

best_value = None
best_population = None

fitness_list = []
indices_list = []

best_population_list = []
cnt = 0
mutation_probability = 0.25

for k in range(2):
    while True:
        best_value = None
        best_population = None
        population = create_population(G)
        for i in range(10000):
            fitness = evaluate_population(G, population)
            new_population = crossover(G, population)
            if random.random() > mutation_probability:
                new_population = evolution_step(new_population)
            fitness_after = evaluate_population(G, new_population)
            # print("old:", population)
            #print("new:", new_population)
            #print("fitness_after:", fitness_after)
            if fitness_after < fitness:
                population = new_population
                fitness = fitness_after
                # print("Evolution successful!")
            #if check_contraints(G, population):
            if not best_population:
                best_population = population
                best_value = fitness
            else:
                if fitness < best_value:
                    best_population = population
                    best_value = fitness
        cnt += 1
        invalid_tours = is_valid_solution(G, best_population)
        if invalid_tours == 0:
            print("*"*20)
            print("Valid Solution found")
            print("*"*20)
            break
        else:
            print("Invalid Solution ({} invalid tours)".format(invalid_tours))
            print("I [{}]: {}".format(cnt, best_value))
    best_population_list.append(best_population)
    
    
best_population = None
best_value = None
for p in best_population_list:
    fitness = evaluate_population(G, p)
    if not best_population:
        best_population = p
        best_value = fitness
    else:
        if fitness < best_value:
            best_population = p
            best_value = fitness
    
print("RESULTS:")           
pretty_print_population(G, best_population)

draw_hwn_map.draw_map(get_routes(best_population))
    
'''
fig, ax = plt.subplots()
ax.plot(indices_list, fitness_list)
ax.set(xlabel='indices', ylabel='fitness',
       title=' ')
ax.grid()
# ax.set_yscale('log')
fig.savefig("test.png")
plt.show()
'''
# draw_graph(G)
