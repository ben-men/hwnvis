import folium
import xmltodict
import os 
import html

gpx_folder = "tours"
BASIC_COLORS_NORMAL = [
    "#d65f5f",
    "#956cb4",
    "#8c613c",
    "#dc7ec0",
    "#797979",
    "#4878d0",
    "#ee854a",
    "#d1345b",
    "#070707",
    "#e85d75",
    "#3bc14a",
    #"#a4243b",
    #"#bd632f",
    #"#8884ff",
    #"#190b28"
]

def draw_map(routes = None, limit = None):
    files = os.listdir(gpx_folder)

    all_track_points = []
    tour_track_points = []
    max_lon = None
    min_lon = None
    max_lat = None
    min_lat = None

    for f in files:
        if not f.endswith(".gpx"):
            continue
        with open(os.path.join(gpx_folder, f), encoding="utf-8") as fd:
            doc = xmltodict.parse(fd.read())
            new_track_points = doc["gpx"]["trk"]["trkseg"]["trkpt"]

            for t in new_track_points:
                # print(t)
                if not max_lon or max_lon<t["@lon"]:
                    max_lon = t["@lon"]
                if not min_lon or min_lon>t["@lon"]:
                    min_lon = t["@lon"]
                if not max_lat or max_lat<t["@lat"]:
                    max_lat = t["@lat"]
                if not min_lat or min_lat>t["@lat"]:
                    min_lat = t["@lat"]
            all_track_points.extend(new_track_points)
            tour_track_points.append(new_track_points)

    center_lat = (float(max_lat) + float(min_lat)) / 2.0
    center_lon = (float(max_lon) + float(min_lon)) / 2.0
        
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

    """
    points = []
    #print(len(track_points))
    for i, t in enumerate(all_track_points):
        marker_color = 'gray'
        caption = str(i)
        #folium.Marker(
        #    location=[float(t["@lat"]), float(t["@lon"])],
        #    icon=folium.Icon(color=marker_color, icon='info-sign'),
        #    popup=caption
        #).add_to(m)
        #
        points.append((float(t["@lat"]), float(t["@lon"])))
    #print(len(points))
    folium.PolyLine(points).add_to(m)
    """

    for i, tour in enumerate(tour_track_points):
        points = []
        for p in tour:
            marker_color = 'gray'
            caption = str(i)
            #folium.Marker(
            #    location=[float(p["@lat"]), float(p["@lon"])],
            #    icon=folium.Icon(color=marker_color, icon='info-sign'),
            #    popup=caption
            #).add_to(m)
            #
            points.append((float(p["@lat"]), float(p["@lon"])))
        print("color=", BASIC_COLORS_NORMAL[i%len(BASIC_COLORS_NORMAL)])
        folium.PolyLine(points, popup=str(i), color=BASIC_COLORS_NORMAL[i%len(BASIC_COLORS_NORMAL)]).add_to(m)
        

    m.save('index.html')
    

if __name__ == "__main__":
    draw_map()