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
from copy import deepcopy
import json
import argparse
import time

hwn_file_name = os.path.join("hwn_gpx", "HWN_2020_05_01.gpx")

def draw_map():
    with open(hwn_file_name, encoding="utf-8") as fd:
        doc = xmltodict.parse(fd.read())
        
        bounds = doc["gpx"]["metadata"]["bounds"]  
        center_lat = (float(bounds["@maxlat"]) + float(bounds["@minlat"])) / 2.0
        center_lon = (float(bounds["@maxlon"]) + float(bounds["@minlon"])) / 2.0
        
        m = folium.Map(location=[center_lat, center_lon])
        
        coord_map = {}
        stamps = doc["gpx"]["wpt"]
        for cnt, s in enumerate(stamps):
            caption = s["name"]
            coord_map[caption] = (float(s["@lat"]), float(s["@lon"]))
            if "desc" in s:
                caption += " "+s["desc"]    
            somebody_was_there = False
            marker_color = 'gray'
            who_was_there = []
                    
            if len(who_was_there) > 0:        
                caption += "\n"+"Visitors: {}".format(",".join(who_was_there))   
            caption = html.escape(caption)
            folium.Marker(
                location=[float(s["@lat"]), float(s["@lon"])],
                icon=folium.Icon(color=marker_color, icon='info-sign'),
                popup=caption
            ).add_to(m)
            
            # <ele>555.0000000</ele>
        """folium.Marker(
                location=[float(mc[0]), float(mc[1])],
                icon=folium.Icon(color="red", icon='info-sign'),
                popup="Most central"
            ).add_to(m)
        """
        m.save('stat.html')

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
            if not "cmt" in s:
                cmt = caption
            else:
                cmt = s["cmt"]
            G.add_node(int(caption.replace("HWN","")), lat=float(s["@lat"]), lon=float(s["@lon"]), alt=float(s["ele"]), visitors_cnt = visitors_cnt, desc=cmt)
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


def get_nearest_stamp(G, for_stamp):
    nearest_dist = None
    nearest_stamp = None
    for n in G.edges([for_stamp]):
        dist = haversine(G.nodes[n[0]]['lon'], G.nodes[n[0]]['lat'], G.nodes[n[1]]['lon'], G.nodes[n[1]]['lat'])
        if nearest_dist is None or dist < nearest_dist:
            nearest_dist = dist
            nearest_stamp = n[1]
    return nearest_stamp, nearest_dist


def init_arg_parser():
    """Initialize the command-line argument parser and adds all possible command line options.

    :return: an initialized command-line parser with all possible options
    """
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json', help='Output result JSON file name', required=False, type=str)
    args = parser.parse_args()

def print_nesw(G):
    # nördlichster/südlichster/...:
    max_coord = 0
    max_n_1 = None
    for n_1 in G.nodes:
        coord = G.nodes[n_1]['lat']
        if max_n_1 is None or coord > max_coord: 
            max_n_1 = n_1
            max_coord = coord
    print("Nördlichster: {} ({})".format(max_n_1, G.nodes[max_n_1]['desc']))
    max_coord = 0
    max_n_1 = None
    for n_1 in G.nodes:
        coord = G.nodes[n_1]['lat']
        if max_n_1 is None or coord < max_coord: 
            max_n_1 = n_1
            max_coord = coord
    print("Südlichster: {} ({})".format(max_n_1, G.nodes[max_n_1]['desc']))
    max_coord = 0
    max_n_1 = None
    for n_1 in G.nodes:
        coord = G.nodes[n_1]['lon']
        if max_n_1 is None or coord > max_coord: 
            max_n_1 = n_1
            max_coord = coord
    print("Östlichster: {} ({})".format(max_n_1, G.nodes[max_n_1]['desc']))
    max_coord = 0
    max_n_1 = None
    for n_1 in G.nodes:
        coord = G.nodes[n_1]['lon']
        if max_n_1 is None or coord < max_coord: 
            max_n_1 = n_1
            max_coord = coord
    print("Westlichster: {} ({})".format(max_n_1, G.nodes[max_n_1]['desc']))
    
def print_isolated(G):
    # isoliertester stempel, d.h. längste luftlinie zum nächsten stempel:
    d_1 = None
    n_1 = None
    n_2 = None
    distances = []
    nodes = []
    for n in G.nodes:
        s, d = get_nearest_stamp(G, n)
        if d_1 is None or d > d_1: 
            d_1 = d
            n_1 = n
            n_2 = s
    print("Isoliertester Stempel: {} ({})), d={}".format(n_1, G.nodes[n_1]['desc'], d_1))

def print_longest_dist(G):
    # längste luftlinie:
    max_dist = 0
    max_n_1 = None
    max_n_2 = None
    for n_1 in G.nodes:
        for n_2 in G.nodes:
            dist = haversine(G.nodes[n_1]['lon'], G.nodes[n_1]['lat'], G.nodes[n_2]['lon'], G.nodes[n_2]['lat'])
            if max_n_1 is None or dist > max_dist: 
                max_n_1 = n_1
                max_n_2 = n_2
                max_dist = dist
    print("Längste Luftlinie: {} ({})-{} ({}), d={}".format(max_n_1, G.nodes[max_n_1]['desc'], max_n_2, G.nodes[max_n_2]['desc'], max_dist))

def print_highest(G):
    # höchster:
    d_1 = None
    n_1 = None
    for n in G.nodes:
        if d_1 is None or G.nodes[n]['alt'] > d_1:  
            d_1 = G.nodes[n]['alt']
            n_1 = n
    print("Höchster: {} ({}), d={}".format(n_1, G.nodes[n_1]['desc'], d_1))
    
def print_lowest(G):
    # tiefster:
    d_1 = None
    n_1 = None
    for n in G.nodes:
        if d_1 is None or G.nodes[n]['alt'] < d_1:  
            d_1 = G.nodes[n]['alt']
            n_1 = n
    print("Tiefster: {} ({}), d={}".format(n_1, G.nodes[n_1]['desc'], d_1))

def print_nearest(G):
    # kürzeste luftlinie:
    d_1 = None
    n_1 = None
    n_2 = None
    for n in G.nodes:
        s, d = get_nearest_stamp(G, n)
        if d_1 is None or d < d_1: 
            d_1 = d
            n_1 = n
            n_2 = s
    print("Kürzeste Luftlinie: {} ({})-{} ({}), d={}".format(n_1, G.nodes[n_1]['desc'], n_2, G.nodes[n_2]['desc'], d_1))

def get_most_central(G):
    sum_lon = 0.0
    sum_lat = 0.0

    for n_1 in G.nodes:
        sum_lon += G.nodes[n_1]['lon']
        sum_lat += G.nodes[n_1]['lat']
    
    mc = (sum_lat/len(G.nodes), sum_lon/len(G.nodes))

    min_dist = None
    min_node = None
    for n in G.nodes:
        dist = haversine(mc[1], mc[0], G.nodes[n]['lon'], G.nodes[n]['lat'])
        if not min_dist or min_dist>dist:
            min_dist = dist
            min_node = n
    print("Zentralster: {} ({})".format(min_node, G.nodes[min_node]['desc']))

if __name__ == '__main__':
    init_arg_parser()
    
    G = prepare_graph()
    
    # TODOs:
    #   meiste stempel in einem umkreis
    # 
    print_highest(G)
    print_lowest(G)
    print_nearest(G)
    print_isolated(G)
    print_longest_dist(G)
    print_nesw(G)
    get_most_central(G)
    draw_map()