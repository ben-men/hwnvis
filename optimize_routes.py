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

hwn_file_name = os.path.join("hwn_gpx", "HWN_2020_05_01.gpx")
done_stamps_folder = "stamps"

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
        return 7.5 # magic!
    tour_length = 0.0
    for i in range(len(tour)):
        from_node = tour[i]
        to_node = tour[(i+1) % len(tour)]
        # print("{} -- {}".format( tour[i], tour[(i+1) % len(tour)]))
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
    to_be_chosen = set(G.nodes)
    already_chosen = set()
    population = []
    while(len(to_be_chosen) > 0):
        tour_length = random.randint(1, 8)
        new_tour = []
        for i in range(tour_length):
            next_choice = random.choice(tuple(to_be_chosen))
            to_be_chosen.remove(next_choice)
            new_tour.append(next_choice)
            if len(to_be_chosen) == 0:
                break
        population.append(new_tour)
    print("population=", population)
    return population
        

def prepare_graph():
    limit = 10
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
            G.add_node(int(caption.replace("HWN","")), lat=float(s["@lat"]), lon=float(s["@lon"]))
            if int(caption.replace("HWN","")) == limit:
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

def evaluate_population(G, population):
    perfect_length = 18.0
    total_len_km = 0.0
    total_derivation_km = 0.0
    total_len_tours = len(population)
    for i, t in enumerate(population):
        tour_length = eval_tour(G, t)
        #print("[{}] tour=".format(i),t)
        #print("[{}] tour_length=".format(i), tour_length)
        total_len_km += tour_length
        total_derivation_km += abs(perfect_length - tour_length) 
    
    fitness = total_derivation_km*total_len_tours  #+ total_len_tours
    return fitness
    
def evolution_step(population):
    # arbitrary positions changing
    """
    affected_tour_index = random.randint(0, len(population)-1)
    from_index = random.randint(0, len(population[affected_tour_index])-1)
    to_index = random.randint(0, len(population[affected_tour_index])-1)
    population[affected_tour_index][from_index], population[affected_tour_index][to_index] = population[affected_tour_index][to_index], population[affected_tour_index][from_index]
    """
    affected_tour_index = random.randint(0, len(population)-1)
    affected_index = random.randint(0, len(population[affected_tour_index])-1)
    other_index = (affected_index+1) % len(population[affected_tour_index])
    population[affected_tour_index][affected_index], population[affected_tour_index][other_index] = population[affected_tour_index][other_index], population[affected_tour_index][affected_index]
    
    return population
    
G = prepare_graph()
#tour = create_random_tour(G)
#tour_length = eval_tour(G, tour)
#print("tour=",tour)
#print("tour_length=",tour_length)
best_value = None
best_population = None
for i in range(50):
    population = create_population(G)
    fitness = evaluate_population(G, population)
    # print("fitness=",fitness)
    if not best_population:
        best_population = population
        best_value = fitness
    else:
        if fitness < best_value:
            best_population = population
            best_value = fitness
            
print("RESULTS:") 
print("pop=",best_population)           
print("fitness=",best_value)

population = evolution_step(best_population)

fitness = evaluate_population(G, population)

print("pop=",population)           
print("fitness=",fitness)
# draw_graph(G)
