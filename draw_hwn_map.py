import folium
import xmltodict
import os 
import html

hwn_file_name = os.path.join("hwn_gpx", "HWN_2020_05_01.gpx")
done_stamps_folder = "stamps"

COLORS = ['gray', "#FFC300", "#FF5733", "#C70039", "#900C3F", "#581845"]

def draw_map(routes = None, limit = None):
    files = os.listdir(done_stamps_folder)
    all_stamps_lists = {}
    total_num_visitors = len(files)
    for f in files:
        if not f.endswith(".txt"):
            continue
        stamps_list = []
        with open(os.path.join(done_stamps_folder, f)) as fd:
            for line in fd:
                stamps_list.append("HWN"+line.strip())
        all_stamps_lists[f.replace(".txt", "")] = (stamps_list, "red")

    with open(hwn_file_name, encoding="utf-8") as fd:
        doc = xmltodict.parse(fd.read())
        
        bounds = doc["gpx"]["metadata"]["bounds"]  
        center_lat = (float(bounds["@maxlat"]) + float(bounds["@minlat"])) / 2.0
        center_lon = (float(bounds["@maxlon"]) + float(bounds["@minlon"])) / 2.0
        
        m = folium.Map(location=[center_lat, center_lon])
        
        coord_map = {}
        stamps = doc["gpx"]["wpt"]
        for cnt, s in enumerate(stamps):
            if limit and cnt >= limit:
                break
            caption = s["name"]
            coord_map[caption] = (float(s["@lat"]), float(s["@lon"]))
            if "desc" in s:
                caption += " "+s["desc"]    
            somebody_was_there = False
            marker_color = 'gray'
            who_was_there = []
            for k, v in all_stamps_lists.items():
                if s["name"] in v[0]:
                    somebody_was_there = True
                    #marker_color = v[1]
                    who_was_there.append(k)
                    
            if len(who_was_there) > 0:
                caption += "\n"+"Visitors: {}".format(",".join(who_was_there))
            marker_color = COLORS[len(who_was_there)]
            caption = html.escape(caption)
            folium.Marker(
                location=[float(s["@lat"]), float(s["@lon"])],
                icon=folium.Icon(color='gray', icon_color=marker_color, icon='info-sign'),
                popup=caption
            ).add_to(m)
            
            # <ele>555.0000000</ele>
        
        if routes:
            for r in routes:
                points = []
                for p in r:
                    hwn_formatted = ("HWN{:03d}".format(p)) 
                    points.append(coord_map[hwn_formatted])
                # Add first point again to make the route a circle
                hwn_formatted = ("HWN{:03d}".format(r[0])) 
                points.append(coord_map[hwn_formatted])
                
                folium.PolyLine(points).add_to(m)

        m.save('index.html')

if __name__ == "__main__":
    draw_map()