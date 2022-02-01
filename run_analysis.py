
from analysis_methods.volume_measurement.analysis import run_volume_measurement
from analysis_methods.pH_measurement.analysis import run_pH_measurement

dir = input ('Data source directory') # data sourse
last = int(input("Last frame intiger number")) # last frame
first = int(input("First frame intiger number")) # first frame

conditions_number = int(input("Number of conditions to be analyzed"))
conditions_entry = input ('Enter condition names separated space') # list of conditions
if conditions_number > 1:
   conditions_list = list(map(str, conditions_entry.split(' ')))
else:
   conditions_list = [conditions_entry]

analysis_type = input ('Type of analusis: enter "volume" or "pH"') # Analyze volume or pH

if analysis_type == 'volume':
   run_volume_measurement(dir, last, first, conditions_list)
elif analysis_type == 'pH':
   background = float(input("Background percentile, for current analysis we use 2"))
   YFP_c = int(input("Number ID of pH-sensitive channel"))
   FRET_c = int(input("Number ID of pH-insensitive channel used for naormalization"))

   run_pH_measurement(dir, last, first, conditions_list, background, YFP_c, FRET_c)