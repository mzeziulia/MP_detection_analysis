
from analysis_methods.volume_measurement.analysis import run_volume_measurement
# from analysis_methods.pH_measurement.analysis import run_pH_measurement

date_dir = input ('Data source directory') # data sourse
last_frame = int(input("Last frame intiger number")) # last frame
first_frame = int(input("First frame intiger number")) # first frame

conditions_entry = input ('Enter condition names separated space') # list of conditions
conditions = list(map(int,conditions_entry.split(' ')))

analysis_type = input ('Type of analusis: enter "volume" or "pH"') # Analyze volume or pH

if analysis_type == 'volume':
   run_volume_measurement(date_dir, last_frame, first_frame, conditions)
# elif analysis_type == 'pH':
#    run_pH_measurement(date_dir, last_frame, first_frame, conditions)