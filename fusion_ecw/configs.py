BOX_INNER = 0.4          # use central 40% area for robust stats
MIN_LIDAR_PTS = 6        # minimum points per box to trust per-box scale
SCALE_CLAMP = (0.3, 3.5) # clamp s to avoid explosions
EMA_ALPHA = 0.3          # temporal smoothing for per-track scale
MED_FILTER_LIDAR = 3     # optional median filter kernel on Dlidar
TTCCLASS_THRESH = {"ped": 2.0, "bike": 2.0, "vehicle": 1.5}
DRAC_THRESH_MPS2 = {"ped": 2.5, "bike": 2.5, "vehicle": 3.5}
RMIN = 0.7               # meters
