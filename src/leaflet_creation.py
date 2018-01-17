# coding: utf8
# layer options: http://leaflet-extras.github.io/leaflet-providers/preview/
MAP_LAYERS = {
    'dark_black': """L.tileLayer('https://cartodb-basemaps-{s}.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png', {attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="http://cartodb.com/attributions">CartoDB</a>',});
    """,
    'osm_mapnik': """L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'});
     """,
}
template = """
<!DOCTYPE html>
<html>
<head>
    <title>{html_title}</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.0.1/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.0.1/dist/leaflet.js"></script>
    <style>
        #map {{
            width: {width}px;
            height: {height}px;
        }}
    </style>
</head>
<body>

<!-- container for the map-->
<div id='map'></div>

<!-- geojson data file-->
<script src="{file_name}.js" type="text/javascript"></script>

<script>
    function set_style(feature){{return {{color: feature.properties.color}};}}

    function onEachFeature(feature,layer){{
        var popUpContent = '';
        for (var key in feature.properties) {{
            val = feature.properties[key];
            popUpContent += key + ':' + val + "<br>";
        }}
        layer.bindPopup(popUpContent);
    }}

    var mbUrl = 'https://api.tiles.mapbox.com/v4/{{id}}/{{z}}/{{x}}/{{y}}.png?access_token=pk.eyJ1Ijoic3VyYWpuYWlyIiwiYSI6ImNpdWoyZGQzYjAwMXkyb285b2Q5NmV6amEifQ.WBQAX7ur2T3kOLyi11Nybw';
    {map_style_layers}
    {bind_data_to_layers}
    {check_layers}
    {radio_layers}
    {map}
    L.control.layers(check_layers, radio_layers).addTo(map);
</script>
</body>
</html>
"""
def gradient_color(percent):
    import numpy as np
    min_color = np.array([251,248,179])
    max_color = np.array([248,105,107])
    return '#%02x%02x%02x' % tuple([int(k) for k in min_color+(max_color-min_color)*percent])

def get_color_for_df(df, cnt_col, log_=True):
    import numpy as np
    df['color'] = df[cnt_col]
    if log_:
        df['color'] = df[cnt_col]+1
        df['color'] = np.log(df['color'])
    df['color'] = df['color']/df['color'].max()
    df['color'] = df['color'].apply(gradient_color)


def get_map(lat, lon, zoom, init_layers):
    map_str = """
    var map = L.map('map', {{
        center: [{lat}, {lon}],
        zoom: {zoom},
        {init_layers}
    }});
    """
    init_layers_str = 'layers: [{}]'.format(', '.join(init_layers)) if init_layers else ''
    return map_str.format(lat=lat, lon=lon, zoom=zoom, init_layers=init_layers_str)


def get_map_style_layers(map_layers=['streets']):
    map_style_str = ''
    for s in map_layers:
        map_style_str += 'var '+s+' = '+MAP_LAYERS[s]+'\n'
    if not map_style_str:
        map_style_str = MAP_LAYERS['dark-black']
    return map_style_str


def get_bind_data_to_layers(binding_data):
    bind_data_str = ''
    for var_name, _ in binding_data:
        bind_data_str += """
    var {v}_layer = new L.LayerGroup();
    L.geoJSON({v}, {{style: set_style,onEachFeature: onEachFeature}}).addTo({v}_layer);
    """.format(v=var_name)
    return bind_data_str


def get_check_radio_layers(binding_data,map_layers):
    choice = {0: 'check_layers', 1: 'radio_layers'}
    check_radio_layers = ['', '']
    for var_name in map_layers:
        check_radio_layers[0] += "'{v}': {v}, ".format(v=var_name)
    for var_name, display_str in binding_data:
        check_radio_layers[1] += "'{d}': {v}_layer, ".format(d=display_str, v=var_name)

    for i, s in enumerate(list(check_radio_layers)):
        if s:
            check_radio_layers[i] = """
    var {0} = {{
        {1}
    }};
            """.format(choice[i], s)
    return check_radio_layers


def clean_init_layers(init_layers, binding_data):
    cleaned_init_layers = []
    binding_data_layers = [b[0] for b in binding_data]
    for il in init_layers:
        if il in MAP_LAYERS:
            cleaned_init_layers.append(il)
        if  il in binding_data_layers:
            cleaned_init_layers.append(il+'_layer')
            
    return cleaned_init_layers


def create_leaflet(html_title, file_path, file_name, lat, lon, zoom, init_layers, map_layers, binding_data, width=700, height=700):

    # allow_style = ['light', 'dark', 'outdoors', 'satellite', 'streets']
    # if len(set(map_layers)-set(allow_style))!=0:
    #     raise ValueError('allow map style layers is %s' % str(allow_style))
    init_layers = clean_init_layers(init_layers, binding_data)
    map = get_map(lat, lon, zoom, init_layers)
    map_style_str = get_map_style_layers(map_layers)
    bind_data_to_layers = get_bind_data_to_layers(binding_data)
    check_radio_layers = get_check_radio_layers(binding_data, map_layers)
    check_layers = check_radio_layers[0]
    radio_layers = check_radio_layers[1]
    with open(file_path+file_name+'.html','w') as f:
        f.write(template.format(html_title=html_title, file_name=file_name, width=width, height=height, map=map,
                      map_style_layers=map_style_str, bind_data_to_layers=bind_data_to_layers,
                          check_layers=check_layers, radio_layers=radio_layers))


def create_js_data(file_path, file_name, binding_data, gpdfs):
    with open(file_path+ file_name+'.js', 'w') as f:
        for i, bd in enumerate(binding_data):
            var = bd[0]
            gpdf = gpdfs[i]
            js = gpdf.to_json()
            f.write('var {var} = {js};\n'.format(var=var, js=js))


def create_map_visualization(html_title, file_path, file_name, lat, lon, zoom,
                             init_layers, map_layers, binding_data, gpdfs, width=700, height=700):
    """
    example:
    html_title = 'html title'
    file_path = ''
    file_name = 'leaflet file'
    lon, lat = -77.018479, 38.913237 #D.C.
    zoom = 13
    map_layers = MAP_LAYERS.keys()
    init_layers = [list(map_layers)[0], 'csl']
    binding_data=[['csl','spots of csl']]
    # gpdf1['color'] = '#aa0'
    # gpdf2['color'] = '#0a0'
    gpdfs = [gpdf]
    create_map_visualization(html_title, file_path, file_name, lat, lon, zoom, init_layers, map_layers, binding_data, gpdfs)   
    """
    assert len(binding_data)==len(gpdfs)
    create_leaflet(html_title, file_path, file_name, lat, lon, zoom, init_layers, map_layers, binding_data, width, height)
    create_js_data(file_path, file_name, binding_data, gpdfs)


def test():
    html_title = 'openstreetmap elements'
    file_path = ''
    file_name = 'test creation of leaflet'
    lon, lat = -77.018479, 38.913237 #D.C.
    zoom = 14
    init_layers = ['streets', 'stsg']
    map_layers = ['light','streets', 'satellite']
    binding_data=[['stsg','street segment'],['stsg1','street segment1']]
    import geopandas as gp
    from shapely.geometry import Point
    gpdfs = []
    gpdfs.append(gp.GeoDataFrame([Point(-77.116761, 38.9305064),Point(-77.1069168, 38.9195066)], columns=['geometry']))
    gpdfs.append(gp.GeoDataFrame([Point(-77.0908494, 38.9045525),Point(-77.0684995, 38.9000923)], columns=['geometry']))
    create_map_visualization(html_title, file_path, file_name, lat, lon, zoom, init_layers, map_layers, binding_data, gpdfs)

if __name__ == "__main__":
    test()