from config import *
import circle_finding_functions as crcl
import contour_and_cell_mask_functions as mask
import plotting_and_parameters_readout_functions as stats
from analysis_methods import run_analysis_X_way, run_analysis_Y_way, run_analysis_Z_way

last_frame = int(input("last_frame"))
first_frame = int(input("first_frame"))

conditions = []
question = 'yes'
i=0

while question == 'yes':
   conditions.append(input('name_of_condition'))
   question =  input("Are there other conditions? Type yes or no")
   i=i+1


if keyword_arg1 == X:
   run_analysis_X_way()
elif keyword_arg2 == Y:
   run_analysis_Y_way()