import json
import os
import draw_hwn_map


if __name__ == '__main__':
    path = "."
    best_length = [None, None, None, None, None]
    best_tours = [None, None, None, None, None]
        
    for json_file in os.listdir(path):
        
        if not json_file.endswith(".json"):
            continue
        with open(os.path.join(path, json_file), "r") as input_file:
            data = json.load(input_file)
            process_time_seconds = data["process_time"]
            print("process_time_seconds=", process_time_seconds)
            for i, p in enumerate(data["populations"]):
                if best_length[0] is None or p["length_km"] < best_length[0]:
                    best_length[0] = p["length_km"]
                    best_length[1] = json_file
                    best_length[2] = i
                    best_length[3] = len(p["tours"])
                    best_length[4] = p["tours"]
                if best_tours[0] is None or len(p["tours"]) < best_tours[0]:
                    best_tours[0] = len(p["tours"])
                    best_tours[1] = json_file
                    best_tours[2] = i
                    best_tours[3] = p["length_km"]
                    best_tours[4] = p["tours"]
            
    print("best_length=", best_length)    
    print("best_tours=", best_tours)  
    
    routes = []
    for t in best_tours[4]:
        points = []
        for p in t["tour"]:
            points.append(p)
        routes.append(points)
            
    draw_hwn_map.draw_map(routes)
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