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
    "#3bc14a"
]

BASIC_COLORS_NORMAL = [
    "#7372af",
    "#B33F00",
    "#bd5296",
    "#662400",
    "#FF6B1A",
    "#006663",
    "#344e5e",
    "#483056",
    "#c84961",
    "#3c75a7"
]

def draw_map(highlight_newest, gradient):
    tour_track_points = []
    max_lon = None
    min_lon = None
    max_lat = None
    min_lat = None

    files = [os.path.join(gpx_folder, f) for f in os.listdir(gpx_folder) if f.endswith(".gpx")] # add path to each file
    files.sort(key=lambda x: os.path.getmtime(x))
    for f in files:
        with open(f, encoding="utf-8") as fd:
            doc = xmltodict.parse(fd.read())
            new_track_points = doc["gpx"]["trk"]["trkseg"]["trkpt"]

            for t in new_track_points:
                if not max_lon or max_lon<t["@lon"]:
                    max_lon = t["@lon"]
                if not min_lon or min_lon>t["@lon"]:
                    min_lon = t["@lon"]
                if not max_lat or max_lat<t["@lat"]:
                    max_lat = t["@lat"]
                if not min_lat or min_lat>t["@lat"]:
                    min_lat = t["@lat"]
            tour_track_points.append(new_track_points)

    center_lat = (float(max_lat) + float(min_lat)) / 2.0
    center_lon = (float(max_lon) + float(min_lon)) / 2.0
        
    from colour import Color
    start_color = Color("#a6fd29")
    end_color = Color("#37013a")
    colors = list(start_color.range_to(end_color, len(tour_track_points)))

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
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
        if highlight_newest:
            if i<len(tour_track_points)-1:
                color = BASIC_COLORS_NORMAL[0]
            else:
                color = BASIC_COLORS_NORMAL[1]
        else:
            if gradient:
                color = colors[i].hex
            else:
                color = BASIC_COLORS_NORMAL[i%len(BASIC_COLORS_NORMAL)]
        folium.PolyLine(points, popup=str(i), color=color).add_to(m)
        

    m.save('index.html')
    

if __name__ == "__main__":
    highlight_newest = False
    gradient = True
    draw_map(highlight_newest, gradient)