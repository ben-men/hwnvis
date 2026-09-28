import folium
import xmltodict
import os 
import html

hwn_file_name = os.path.join("hwn_gpx", "HWN_2020_05_01.gpx")
done_stamps_folder = "stamps"

COLORS = ['gray', "#FFC300", "#FF5733", "#C70039", "#900C3F", "#581845"]
#HIKER_COLORS = ['#0000fe', '#ff00ff', '#fe0000', '#ffff01', '#00ff01', '#01ffff', '#ffffff']
AVAILABLE_HIKER_COLORS = ['#0000fe', '#fe0000', '#00ff01', '#01ffff', '#ffffff', '#ff00ff', '#ffff01']

COLOR_BY_NUM = False

# Funktion zum Mischen von Farben im Hex-Format
def hex_to_rgb(hex_color):
    """Konvertiert einen Hex-Farbcode in RGB."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb_color):
    """Konvertiert eine RGB-Farbe in einen Hex-Farbcode."""
    return '#' + ''.join(f'{x:02x}' for x in rgb_color)

def mix_colors(*colors):
    """Mischt mehrere Hex-Farben additiv und gibt das Ergebnis als Hex zurück."""
    total_r, total_g, total_b = 0, 0, 0
    num_colors = len(colors)
    if num_colors==0:
        return '#000000'

    # Addiere die RGB-Werte jeder Farbe
    for color in colors:
        r, g, b = hex_to_rgb(color)
        total_r += r
        total_g += g
        total_b += b

    # Berechne den Durchschnitt (sollte mit 255 limitiert werden)
    avg_r = min(total_r // num_colors, 255)
    avg_g = min(total_g // num_colors, 255)
    avg_b = min(total_b // num_colors, 255)

    return rgb_to_hex((avg_r, avg_g, avg_b))

def draw_map(routes = None, limit = None):
    files = os.listdir(done_stamps_folder)
    all_stamps_lists = {}
    total_num_visitors = len(files)
    hiker_colors = {}
    for f in files:
        if not f.endswith(".txt"):
            continue
        stamps_list = []
        with open(os.path.join(done_stamps_folder, f)) as fd:
            for line in fd:
                stamps_list.append("HWN"+line.strip())
        all_stamps_lists[f.replace(".txt", "")] = (stamps_list, "red")
        hiker_colors[f.replace(".txt", "")] = AVAILABLE_HIKER_COLORS[len(hiker_colors)]

    with open(hwn_file_name, encoding="utf-8") as fd:
        doc = xmltodict.parse(fd.read())
        
        bounds = doc["gpx"]["metadata"]["bounds"]  
        center_lat = (float(bounds["@maxlat"]) + float(bounds["@minlat"])) / 2.0
        center_lon = (float(bounds["@maxlon"]) + float(bounds["@minlon"])) / 2.0
        
        m = folium.Map(
            location=[center_lat, center_lon],
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
            attr="Sources: Esri, HERE, Garmin, USGS, Intermap, INCREMENT P, NRCan, Esri Japan, METI, Esri China (Hong Kong), Esri Korea, Esri (Thailand), NGCC, (c) OpenStreetMap contributors, and the GIS User Community",
        )
        
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
            if COLOR_BY_NUM:
                marker_color = COLORS[len(who_was_there)]
            else:
                colors = []
                for who in who_was_there:
                    colors.append(hiker_colors[who])
                marker_color = mix_colors(*colors)

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