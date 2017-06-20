# all fn_ variable below is prefixed with dir_data
dir_data= 'data/'

# ###################################################################### =========================
# ###################################################################### Philly
# ###################################################################### =========================
# CRS
# epsg_ph = 32617  # Kiran said it is the code saved in qgis project
epsg_ph = 3559  # I find this better in buffer 10 meter
# OSM
osm_ph_poly_rid = 188022
fn_city_poly_ph = 'city_poly_ph.geojson'
fn_city_poly_ph_png = fn_city_poly_ph.replace('geojson', 'png')
fn_osm_raw_ph = 'raw_data/osm_raw_ph.osm'
fn_osm_db_ph = 'osm_db_ph.sqlite3'
# Segments
fn_segments_ph = 'segments_ph.geojson'
var_directionality_column_ph = 'ONEWAY'
# FourSquare Venues
dir_frsq_raw_venues_ph = 'raw_data/frsq_raw_venues/ph/'
fn_frsq_venues_ph = 'frsq_venues_ph.geojson'
# Poi
fn_poi_frsq_ph = 'poi_frsq_ph.geojson'
fn_poi_osm_ph = 'poi_osm_ph.geojson'
fn_poi_distr_ph = 'poi_distr_ph.csv'
fn_poi_seg_cvrg_ph = 'poi_seg_coverage_ph.html'
fn_poi_boxplot_per_seg_ph = 'poi_boxplot_per_seg_ph.html'
fn_feature_poi_ph = 'feature_poi_ph.csv'
# Bike facilities
fn_feature_bk_facs_ph = 'feature_bk_facs_ph.csv'
# network feature
# fn_intxn_ph = 'intxn_ph.csv'
fn_feature_seg_as_node_ph = 'feature_seg_as_node_ph.csv'
fn_feature_seg_as_edge_ph = 'feature_seg_as_edge_ph.csv'


# ###################################################################### =========================
# ###################################################################### D.C.
# ###################################################################### =========================
# CRS
epsg_dc = 3559
# OSM
osm_dc_poly_rid = 162069
fn_city_poly_dc = 'city_poly_dc.geojson'
fn_city_poly_dc_png = fn_city_poly_dc.replace('geojson', 'png')
fn_osm_raw_dc = 'raw_data/osm_raw_dc.osm'
fn_osm_db_dc = 'osm_db_dc.sqlite3'
# Segments
fn_segments_dc_raw = 'raw_data/segments_dc_opendc.geojson'
fn_segments_dc = 'segments_dc.geojson'
var_directionality_column_dc = 'DIRECTIONALITY'
# FourSquare Venues
dir_frsq_raw_venues_dc = 'raw_data/frsq_raw_venues/dc/'
fn_frsq_venues_dc = 'frsq_venues_dc.geojson'
# Poi
fn_poi_frsq_dc = 'poi_frsq_dc.geojson'
fn_poi_osm_dc = 'poi_osm_dc.geojson'
fn_poi_distr_dc = 'poi_distr_dc.csv'
fn_poi_seg_cvrg_dc = 'poi_seg_coverage_dc.html'
fn_poi_boxplot_per_seg_dc = 'poi_boxplot_per_seg_dc.html'
# intersection file
fn_intxn_dc = 'intxn_dc.csv'
# opendata.dc.gov
fn_oepndc_bk_dc = 'raw_data/opendc/Bike_Lane_Street_RightofWay.csv'
fn_311_dc = 'raw_data/opendc/Cityworks_Service_Requests.csv'
fn_crash_dc = 'raw_data/opendc/Crashes_in_the_District_of_Columbia.csv'
fn_vision0_dc = 'raw_data/opendc/Vision_Zero_Safety_dc.csv'
fns_crime_dc = {
    2014: 'raw_data/opendc/Crime_Incidents_in_2014.geojson',
    2015: 'raw_data/opendc/Crime_Incidents_in_2015.geojson',
    2016: 'raw_data/opendc/Crime_Incidents_in_2016.geojson',
    2017: 'raw_data/opendc/Crime_Incidents_in_2017.geojson',
}

# features for dc
fn_feature_seg_attr_dc = 'feature_seg_attribute_dc.csv'
fn_feature_poi_dc = 'feature_poi_dc.csv'
fn_feature_bk_facs_dc = 'feature_bk_facs_dc.csv'   # bike facility from osm
fn_feature_seg_as_node_dc = 'feature_seg_as_node_dc.csv'
fn_feature_seg_as_edge_dc = 'feature_seg_as_edge_dc.csv'
fn_feature_crash_dc = 'feature_crash_dc.csv'
fn_feature_311_dc = 'feature_311_dc.csv'
fn_feature_vision0_dc = 'feature_vision0_dc.csv'
fn_feature_crime_dc = 'feature_crime_dc.csv'
fn_feature_mov_dc = 'feature_mov_violations_dc.csv'
fn_feature_parking_dc = 'feature_parking_violations_dc.csv'
fn_feature_oepndc_bk_dc = 'feature_bk_opendc_dc.csv'

# features that are meaningful to have a "total" count column.
features_for_total = ['crash', '311', 'crime', 'v0', 'moving', 'parking', 'poi', 'bk_opendc']

fn_features_dc = {
    'seg_attr': fn_feature_seg_attr_dc,
    'poi': fn_feature_poi_dc, 
    'bk_osm': fn_feature_bk_facs_dc, 
    'net_SaN': fn_feature_seg_as_node_dc, 
    'net_SaE': fn_feature_seg_as_edge_dc, 
    'crash': fn_feature_crash_dc,
    '311': fn_feature_311_dc, 
    'v0': fn_feature_vision0_dc, 
    'crime': fn_feature_crime_dc, 
    'moving': fn_feature_mov_dc, 
    'parking': fn_feature_parking_dc, 
    'bk_opendc': fn_feature_oepndc_bk_dc}

fn_target_lts_dc = 'feature_lts_dc.csv'

# ====================================================================== =========================
# ###################################################################### non city specific constants
# ====================================================================== =========================
# ###################################################################### crs
latlon_crs = 4326
# ###################################################################### geom_helper
index_seg = 'index_seg'
index_obj = 'index_obj'
index_ln = 'index_ln'
index_pt = 'index_pt'
index_from = 'index_f'  # directed intersection network: from segment, to segment
index_to = 'index_t'
index_from_start_point = 'intx_f_start_point'  # intersected at start/end point of from segment
index_from_end_point = 'intx_f_end_point'
index_to_start_point = 'intx_t_start_point'  # intersected at start/end point of to segment
index_to_end_point = 'intx_t_end_point'

# ###################################################################### FourSquare Taxonomy
fn_frsq_taxonomy_json = 'frsq_taxonomy_raw.json'
fn_frsq_taxonomy_csv = 'frsq_taxonomy_parsed.csv'
fn_frsq_taxonomy_tree = 'frsq_taxonomy_tree.txt'

# ###################################################################### Poi mapping
fn_mapping_for_fs = 'manual/poi_mapping_for_fs.txt'
var_exclude_category_for_fs = 'no category'
fn_mapping_for_osm = 'manual/poi_mapping_for_osm.txt'
var_exclude_category_for_osm = 'exclude'
similar_name_threshold = 0.8  # used in detecting frsq and osm duplicates
no_name_value = -1.0
poi_categories = ['art', 'outdoors and recreation', 'retail shop', 'professional service', 'food',
                  'nightlife spot', 'residence', 'schools&university', 'cycling facilities', 'transportation']

# ###################################################################### Network features
# #################################################### segments as edges
ftr_name_d_btw_cntr_SgAsEg = 'd_btw_cntr_SgAsEg'
ftr_name_ud_btw_cntr_SgAsEg = 'ud_btw_cntr_SgAsEg'
ftr_name_ud_bridge_SgAsEg = 'ud_bridge_SgAsEg'
# #################################################### segments as nodes
ftr_name_d_in_deg_SgAsNd = 'd_in_deg_SgAsNd'
ftr_name_d_out_deg_SgAsNd = 'd_out_deg_SgAsNd'
ftr_name_d_node_ecc_SgAsNd = 'd_node_ecc_SgAsNd'
ftr_name_d_clo_cntr_SgAsNd = 'd_clo_cntr_SgAsNd'
ftr_name_d_far_cntr_SgAsNd = 'd_far_cntr_SgAsNd'
ftr_name_d_btw_cntr_SgAsNd = 'd_btw_cntr_SgAsNd'
ftr_name_d_page_rank_SgAsNd = 'd_page_rank_SgAsNd'
ftr_name_d_hub_score_SgAsNd = 'd_hub_score_SgAsNd'
ftr_name_d_auth_score_SgAsNd = 'd_auth_score_SgAsNd'
ftr_name_ud_deg_cntr_SgAsNd = 'ud_deg_cntr_SgAsNd'
ftr_name_ud_node_ecc_SgAsNd = 'ud_node_ecc_SgAsNd'
ftr_name_ud_clo_cntr_SgAsNd = 'ud_clo_cntr_SgAsNd'
ftr_name_ud_far_cntr_SgAsNd = 'ud_far_cntr_SgAsNd'
ftr_name_ud_eig_cntr_SgAsNd = 'ud_eig_cntr_SgAsNd'
ftr_name_ud_btw_cntr_SgAsNd = 'ud_btw_cntr_SgAsNd'
ftr_name_ud_page_rank_SgAsNd = 'ud_page_rank_SgAsNd'
ftr_name_ud_hub_score_SgAsNd = 'ud_hub_score_SgAsNd'
ftr_name_ud_auth_score_SgAsNd = 'ud_auth_score_SgAsNd'
ftr_name_ud_art_pt_SgAsNd = 'ud_art_pt_SgAsNd'
ftr_name_ud_bridge_SgAsNd = 'ud_bridge_SgAsNd'

# ###################################################################### Bike facilities
# TODO: highway='footway', footway=sidewalk
tag_for_pattern = ['highway', 'cycleway', 'cycleway:left', 'cycleway:right', 'cycleway:both',
                   'oneway:bicycle', 'bicycle', 'bicycle:lanes', 'bicycle:backward',
                   'amenity', 'foot', 'sidewalk', 'oneway', 'lanes']
ftr_name_bk_facs = ['cycle_lane', 'is_shared', 'cycle_way', 'side_walk', 'bikable']

# used in producing bk facilities patterns, in developing bk facs assignment rules.ipynb
bk_type = {
    'L1a_1': [('highway', '*'), ('cycleway', 'lane')],
    'L1a_2': [('highway', '*'), ('cycleway:left', 'lane'), ('cycleway:right', 'lane'), ],
    'L1a_3': [('highway', '*'), ('cycleway:both', 'lane'), ],
    'L1b_1': [('highway', '*'), ('cycleway:right', 'lane'), ('oneway:bicycle', 'no')],
    'L1b_2': [('highway', '*'), ('cycleway', 'lane')],
    'L2': [('highway', '*'), ('cycleway:right', 'lane'), ('-oneway:bicycle', 'no')],
    'M1_1': [('highway', '*'), ('oneway', 'yes'), ('cycleway', 'lane'), ('oneway:bicycle', 'no')],
    'M1_2': [('highway', '*'), ('oneway', 'yes'), ('cycleway:left', 'opposite_lane'), ('cycleway:right', 'lane')],
    'M2a_1': [('highway', '*'), ('oneway', 'yes'), ('cycleway:right', 'lane')],
    'M2a_2': [('highway', '*'), ('oneway', 'yes'), ('cycleway', 'lane')],
    'M2b_1': [('highway', '*'), ('oneway', 'yes'), ('cycleway:left', 'lane'), ('-oneway:bicycle', 'no')],
    'M2b_2': [('highway', '*'), ('oneway', 'yes'), ('cycleway', 'lane')],
    'M2c': [('highway', '*'), ('oneway', 'yes'), ('cycleway', 'lane'), ('lanes', ['2', 2])],
    'M2d': [('highway', '*'), ('oneway', 'yes'), ('cycleway:left', 'lane'), ('oneway:bicycle', 'no'), ],
    'M3a_1': [('highway', '*'), ('oneway', 'yes'), ('oneway:bicycle', 'no'), ('cycleway:left', 'opposite_lane')],
    'M3a_2': [('highway', '*'), ('oneway', 'yes'), ('oneway:bicycle', 'no'), ('cycleway', 'opposite_lane')],
    'M3b_1': [('highway', '*'), ('oneway', 'yes'), ('oneway:bicycle', 'no'), ('cycleway:right', 'opposite_lane')],
    'M3b_2': [('highway', '*'), ('oneway', 'yes'), ('oneway:bicycle', 'no'), ('cycleway', 'opposite_lane')],
    'M4_1': [('highway', '*'), ('oneway', 'yes'), ('cycleway:right', 'lane')],
    'M4_2': [('highway', '*'), ('oneway', 'yes'), ('cycleway', 'lane')],
    'M4_3': [('highway', '*'), ('cycleway', 'lane')],
    'M4_4': [('highway', '*'), ('cycleway:left', 'lane'), ('cycleway:right', 'lane'), ],
    'M4_5': [('highway', '*'), ('cycleway:both', 'lane'), ],

    'T1_1': [('highway', '*'), ('bicycle', 'use_sidepath')],
    'T1_2': [('highway', 'cycleway'), ('oneway', 'yes')],
    'T1_3': [('highway', '*'), ('cycleway', 'track')],
    'T2_1': [('highway', '*'), ('bicycle', 'use_sidepath')],
    'T2_2': [('highway', 'cycleway'), ('oneway', 'no')],
    'T2_3': [('highway', '*'), ('cycleway:right', 'track')],

    'T3_1': [('highway', '*'), ('bicycle', 'use_sidepath')],
    'T3_2': [('highway', 'cycleway'), ('oneway', 'no')],
    'T3_3': [('highway', '*'), ('oneway', 'yes'), ('cycleway:right', 'track'), ('oneway:bicycle', 'no')],

    'T4_1': [('highway', '*'), ('bicycle', 'use_sidepath')],
    'T4_2': [('highway', 'cycleway'), ('oneway', 'yes')],
    'T4_3': [('highway', '*'), ('cycleway:right', 'track')],

    'S1_1': [('highway', '*'), ('oneway', 'yes'), ('oneway:bicycle', 'no')],
    'S1_2': [('highway', '*'), ('oneway', 'yes'), ('cycleway', 'opposite')],

    'foot': [('highway', 'footway'), ],
    'pedestrian': [('highway', 'pedestrian'), ],
    'sidewalk': [('sidewalk', '*'), ],
    'M2d_my': [('highway', 'service'), ('cycleway:left', 'opposite_lane'), ('oneway', 'yes')],
    'M1_my': [('highway', 'secondary'), ('cycleway', 'opposite_lane'), ('bicycle', 'designated'), ('oneway', 'yes'),
              ('lanes', '1')]

}
