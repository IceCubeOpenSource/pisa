import glob, sys, os, tables, collections

import numpy as np


#
# i3 -> PISA variable mapping
#

'''
This class is used to map between I3 frame objects to PISA variables
'''

class I3ToPISAVariableMap() :

	#Sub-class for holding a single variable mapping
	class Mapping() :
		def __init__(self,i3_frame_obj,i3_frame_obj_variable,pisa_variable) :
			self.i3_frame_obj = i3_frame_obj
			self.i3_frame_obj_variable = i3_frame_obj_variable
			self.pisa_variable = pisa_variable


	def __init__(self) :
		self.variables = []


	def add(self,i3_frame_obj,i3_frame_obj_variable,pisa_variable) :
		#TODO Check no duplication in pisa_variable
		self.variables.append( self.Mapping(i3_frame_obj,i3_frame_obj_variable,pisa_variable) )


	def get_all_i3_frame_objs(self) :
		return list(set([ v.i3_frame_obj for v in self.variables ]))


	def get_all_pisa_variables(self) :
		return [ v.pisa_variable for v in self.variables ]


	def __getitem__(self,i3_frame_obj) :
		return [ v for v in self.variables if v.i3_frame_obj == i3_frame_obj ]




#
# i3 -> IceCube HDF5 conversion
#

#Define a function to produce an IceCube-format HDF5 file from a list of input .i3 files
#Uses the I3ToPISAVariableMap class to know which frame objects are required
def i3_to_icecube_hdf5(	i3_files,
						output_file,
						variable_map,
						sub_event_streams=None) :

	from icecube import icetray
	from I3Tray import I3Tray
	from icecube.tableio import I3TableWriter, I3TableService
	from icecube.hdfwriter import I3HDFTableService 

	#from icecube import genie_icetray #TODO toggle GENIE requirement based on need


	#
	# Get inputs
	#

	#Get a list of all i3 frame objects in the mapping
	i3_frame_objs = variable_map.get_all_i3_frame_objs()

	#Set a default sub event stream
	if sub_event_streams is None :
		sub_event_streams = ["InIceSplit"]


	#
	# Perform conversion
	#

	#Create an IceTray instance
	tray = I3Tray()

	#Prepare the HDF5 output table writer
	output_hdf5 = I3HDFTableService(output_file)

	#Add module to parse input i3 files
	tray.AddModule('I3Reader','reader',filenamelist=i3_files)

	#def print_frame(frame) : print frame
	#tray.AddModule(print_frame)

	#Skip frames that do not contain ALL the required i3 frame objects
	def check_all_frame_objects_exist(frame) :
		for var in i3_frame_objs :
			if var not in frame :
				return False
		return True
	tray.AddModule(check_all_frame_objects_exist,"check_frame_objects")

	#Add the HDF5 writer module
	tray.AddModule(I3TableWriter,'writer',
	               tableservice = output_hdf5,
	               SubEventStreams = sub_event_streams,
	               keys = i3_frame_objs
	              )

	#Run it all
	tray.Execute()



#
# pytables helper functions
#

def create_or_append_to_array(hdf5_file,group,array_name,array_data,description=None) :

	#Check if the array already exists
	if array_name in group :

		#It already exists, so append to it
		array = get_attr(group,array_name)
		array.append(array_data)

	else :

		#It doesn't exist yet, create it
		if description is None : description = ""
		hdf5_file.create_array(group, array_name, array_data, description) #TODO Add description field (optional in I3ToPISAVariableMapping)?? 





#
# Main conversion function
#

'''
This function does most of the heavy lifting.
For each category of input data, the IceCube HDF5 writer is used to generate 
an IceCube-format HDF5 file from the input i3 files.
These various HDF% files are then combined into a single PISA-format HDF5
file, including mapping the variable names and conversion to numpy arrays.
'''

def convert_i3_to_pisa(input_data,output_file,variable_map,metadata=None) :


	#
	# Check inputs
	#

	#TODO...

	#input_data must be a dict where:
	#  key = name of a catgory of data/events
	#  value = list of input files for that data category
	#TODO


	#
	# Initialise the output PISA events file
	#

	#Create the file
	output_pisa_events_file = tables.open_file(output_file, mode="w", title="PISA HDF5 Events File")

	#Add some metadata
	#Store it as attributes of the top-level node
	#output_pisa_events_file.root._v_attrs.foo = "bar" 
	if metadata is not None :
		for key,val in metadata.items() :
			setattr(output_pisa_events_file.root._v_attrs,key,val) 
	#TODO What else to store here?


	#
	# Loop over data categories
	#

	i3_frame_objs = variable_map.get_all_i3_frame_objs()

	for data_category,input_files in input_data.items() :

		num_files = len(input_files)
		print "Processing %s : %i files" % (data_category,num_files)


		#
		# Convert i3 files to IceCube-format HDF5
		#

		tmp_hdf5_file_path = "./tmp_%s.hdf5" % data_category

		i3_to_icecube_hdf5(	i3_files=input_files,
							output_file=tmp_hdf5_file_path,
							variable_map=variable_map)


		#
		# Add the data to the top-level output HDF5 file
		#

		#Load the temporary i3 file
		tmp_hdf5_file = tables.openFile(tmp_hdf5_file_path)

		#TODO Check if any tables were acutally written (if user mistypes a frame object name, then nothing is written, need to handle this error better)

		#Create a group for this data_category in the output file
		#data_category_group = output_pisa_events_file.create_group(events_group, data_category, data_category)
		data_category_group = output_pisa_events_file.create_group(output_pisa_events_file.root, data_category, data_category)

		#Loop over variables
		for i3_frame_obj in i3_frame_objs :

			#Check the I3 HDF5 table writer did actually add this frame object to the file
			if not hasattr(tmp_hdf5_file.root,i3_frame_obj) :
				raise Exception("Expected frame object '%s' was not written to the i3 HDF5 file : %s"%(i3_frame_obj,tmp_hdf5_file_path))

			#Get the table correspondig to this variable in tmp hdf5 file
			i3_frame_obj_table = getattr(tmp_hdf5_file.root,i3_frame_obj)

			#Get all mappings for this i3 variable
			mappings = variable_map[i3_frame_obj]

			#Loop through mappings
			for mapping in mappings :

				#Check frame object variable is present #TODO DO this earlier???
				if mapping.i3_frame_obj_variable not in i3_frame_obj_table.colnames :
					raise Exception("Error : Frame object %s does not contain the variable %s" % (i3_frame_obj,mapping.i3_frame_obj_variable) ) 

				#Get the variable and add it to the output PISA HDF5 file group as an array
				i3_frame_obj_array = i3_frame_obj_table.col(mapping.i3_frame_obj_variable)
				create_or_append_to_array( output_pisa_events_file, data_category_group, mapping.pisa_variable, i3_frame_obj_array)

		#Store run and event IDs
		#These are stored for each event for each table, so parse them from any fraem obekct table
		#Get the variable and add it to the output PISA HDF5 file group
		run_id_array = getattr(tmp_hdf5_file.root,i3_frame_objs[0]).col("Run")
		create_or_append_to_array( output_pisa_events_file, data_category_group, "run_id", run_id_array)
		event_id_array = getattr(tmp_hdf5_file.root,i3_frame_objs[0]).col("Event")
		create_or_append_to_array( output_pisa_events_file, data_category_group, "event_id", event_id_array)

		#Store number of files
		num_events = getattr(tmp_hdf5_file.root,i3_frame_objs[0]).nrows
		output_pisa_events_file.create_array(data_category_group, "n_files", np.full(num_events,num_files), "")

		#Close the tmp file
		tmp_hdf5_file.close()


	#
	# Check the resulting data structure
	#

	#TODO num events, all pisa_variables have been created, etc...


	#
	# Done
	#

	output_pisa_events_file.close()


	#TODO Free memory?

	#TODO Is this an efficient way to handle memory in HDF5 ?

	#TODO Delete tmp files?




