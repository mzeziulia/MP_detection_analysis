from config import *
import circle_finding_functions as crcl
import contour_and_cell_mask_functions as mask
import plotting_and_parameters_readout_functions as stats

last_frame = int(input("last_frame"))
first_frame = int(input("first_frame"))

conditions = []
question = 'yes'
i=0

while question == 'yes':
   conditions.append(input('name_of_condition'))
   question =  input("Are there other conditions? Type yes or no")
   i=i+1