#!/usr/bin/env python

'''
Convert GRECO event sample from .i3 files to PISA event files
Tom Stuttard
'''

#TODO Enforce correct environment (e.g. IceCube, not PISA)


"""Conversion factor: convert from centimeters^2 to meters^2"""
CMSQ_TO_MSQ = 1.0e-4



#
# Testing functions
#

#Generate dummy HDF5 file (for testing)
def generate_dummy_hdf5_file(output_file,variable_map,num_events=10) :
	hdf5_file = tables.open_file(output_file,mode="w")
	#events_group = hdf5_file.create_group(hdf5_file.root,"events")
	for nu_key,pdf_code in zip(["nue","numu","nutau"],[12,14,16]) :
		#nu_group = hdf5_file.create_group(events_group,nu_key)
		nu_group = hdf5_file.create_group(hdf5_file.root,nu_key)
		for pisa_var in variable_map.get_all_pisa_variables() :
			if pisa_var == "pdg_code" : fake_data = np.random.choice([-1,1],size=(num_events)) * pdf_code
			elif pisa_var == "interaction" : fake_data = np.random.choice([0,1],size=(num_events))
			else : fake_data = np.random.uniform(0.,10.,size=(num_events))
			array = hdf5_file.create_array(nu_group,pisa_var,fake_data,"")
	hdf5_file.close()


#
# Post-processing functions
#

'''
Combine the pegleg track and cascade hypothesis energies
'''
def combine_reco_track_and_cascade_hypotheses(hdf5_events_file) :

	#Loop over groups
	#for group in hdf5_events_file.root.events :
	for group in hdf5_events_file.root :

		#Combine cascade and track energy
		reco_energy = group.reco_energy_cascade.read() + group.reco_energy_track.read()

		#Replace the indidivual reco energies with this in the table
		hdf5_events_file.create_array(group, "reco_energy", reco_energy, "") 
		hdf5_events_file.remove_node(group, "reco_energy_cascade") 
		hdf5_events_file.remove_node(group, "reco_energy_track")


'''
Split neutrino events by nu vs nubar, and CC vs NC
'''
def subdivide_neutrino_groups(hdf5_events_file) :

	#Loop over groups
	neutrino_groups_found = []
	#for group in hdf5_events_file.root.events :
	for group in hdf5_events_file.root :

		#Select only neutrino groups
		group_name = group._v_name #TODO How to "officialy" get name?
		if group_name.startswith("nu") :

			neutrino_groups_found.append(group_name)

			#Get nubar mask by checking PDG code
			if "pdg_code" not in group :
				raise Exception( "Could not find PDG code for %s" % group_name )
			nubar_mask = group.pdg_code.read() < 0

			#Get CC mask by checking interaction code
			if "interaction" not in group :
				raise Exception( "Could not find interaction code for %s" % group_name )
			cc_mask = group.interaction.read() == 1 #TODO Check code

			#Create new groups
			#nu_cc_group = hdf5_events_file.create_group(hdf5_events_file.root.events,group_name+"_cc")
			#nu_nc_group = hdf5_events_file.create_group(hdf5_events_file.root.events,group_name+"_nc")
			#nubar_cc_group = hdf5_events_file.create_group(hdf5_events_file.root.events,group_name+"bar_cc")
			#nubar_nc_group = hdf5_events_file.create_group(hdf5_events_file.root.events,group_name+"bar_nc")
			nu_cc_group = hdf5_events_file.create_group(hdf5_events_file.root,group_name+"_cc")
			nu_nc_group = hdf5_events_file.create_group(hdf5_events_file.root,group_name+"_nc")
			nubar_cc_group = hdf5_events_file.create_group(hdf5_events_file.root,group_name+"bar_cc")
			nubar_nc_group = hdf5_events_file.create_group(hdf5_events_file.root,group_name+"bar_nc")

			#Fill new groups
			for array in group :
				array_data = array.read()
				hdf5_events_file.create_array(nu_cc_group,array.name,array_data[(~nubar_mask)&cc_mask])
				hdf5_events_file.create_array(nu_nc_group,array.name,array_data[(~nubar_mask)&(~cc_mask)])
				hdf5_events_file.create_array(nubar_cc_group,array.name,array_data[nubar_mask&cc_mask])
				hdf5_events_file.create_array(nubar_nc_group,array.name,array_data[nubar_mask&(~cc_mask)])

			#Remove the old group
			#hdf5_events_file.remove_node(hdf5_events_file.root.events,group_name,recursive=True)
			hdf5_events_file.remove_node(hdf5_events_file.root,group_name,recursive=True)

	if len(neutrino_groups_found) == 0 :
		raise ValueError("Did not find any neutrino groups, cannpt subdivide them")

'''
Calculate the effective area weights
'''
def calc_effective_area_weight(hdf5_events_file) :

	#Loop over groups
	#for group in hdf5_events_file.root.events :
	for group in hdf5_events_file.root :

		#Select only neutrino groups
		group_name = group._v_name #TODO How to "officially" get name?
		if group_name.startswith("nu") :

			#Get neutrino vs antoneutrino generation ratio
			#TODO This needs to be stored in a file somewhere, here using the known value for our GENIE datasets but this is very very nasty
			#TODO This needs to be stored in a file somewhere, here using the known value for our GENIE datasets but this is very very nasty
			#TODO This needs to be stored in a file somewhere, here using the known value for our GENIE datasets but this is very very nasty
			#TODO This needs to be stored in a file somewhere, here using the known value for our GENIE datasets but this is very very nasty
			nu_nubar_genratio = 0.7

			#Calculate weighted effective area
			#Note the conversion from cm^2 to m^2
			weighted_aeff = CMSQ_TO_MSQ * group.one_weight.read() / ( group.n_files.read() * group.n_events.read() * nu_nubar_genratio )
			hdf5_events_file.create_array(group,"weighted_aeff",weighted_aeff,"")

		else :
			raise Exception( "Effective area calculation not yet implemented for '%s'" % group_name )



#
# Main function
#

if __name__ == "__main__" :

	import glob, datetime, os, tables
	import numpy as np

	start_time = datetime.datetime.now()

	#from pisa.utils.i3_to_pisa import I3ToPISAVariableMap, convert_i3_to_pisa
	from i3_to_pisa import I3ToPISAVariableMap, convert_i3_to_pisa
	#from i3_to_pisa_tables import I3ToPISAVariableMap, convert_i3_to_pisa, convert_i3_to_pisa

	debug = True
	dummy_data = False


	#
	# Define input data
	#

	#TODO Use latest GRECO, but need to get correct pegleg variable names...

	#Define inputs files for each category of events...

	#Create container
	neutrinos = {"nue":12,"numu":14,"nutau":16}
	input_data = { cat:[] for cat in neutrinos.keys() } #+["muons","noise"] } 

	#Define GENIE datasets
	greco_top_dir = "/data/ana/LE/NBI_nutau_appearance/level7_24Nov2015"
	genie_dir = os.path.join( greco_top_dir, "genie" )
	genie_dataset = 600
	for nu_key,pdg_code in neutrinos.items() :
		input_data[nu_key].extend( sorted(glob.glob( os.path.join(genie_dir,"%i%i"%(pdg_code,genie_dataset),"*.i3*") )) )
	
	#Define NuGen datasets

	#Define MuonGun datasets

	#Define CORSIKA datasets

	#Define noise datasets

	#Get files


	#Truncate input data if debugging
	if debug :
		max_num_files = 1
		print "WARNING: Truncating number of files in each category to %i in debug mode" % max_num_files
		for k in input_data.keys() :
			input_data[k] = input_data[k][:max_num_files]


	#
	# Define mapping of variables from i3 to PISA
	#

	#Map variables
	variable_map = I3ToPISAVariableMap()

	#TODO Make define some common mappings, and make a way to combine mappings instances?
	#TODO Would it be more generic to make IceTray modules that create the required variables?
	#TODO Could do something like define name of I3Particle for reco, truth particle, etc 

	variable_map.add("I3EventHeader","time_start_utc_daq","time_start_utc_daq")

	variable_map.add("Pegleg_Fit_MNHDCasc","energy","reco_energy_cascade")
	variable_map.add("Pegleg_Fit_MNTrack","energy","reco_energy_track")
	variable_map.add("Pegleg_Fit_MNTrack","zenith","reco_coszen")
	variable_map.add("Pegleg_Fit_MNTrack","length","reco_tracklength")
	variable_map.add("Pegleg_Fit_MNTrack","x","reco_x")
	variable_map.add("Pegleg_Fit_MNTrack","y","reco_y")
	variable_map.add("Pegleg_Fit_MNTrack","z","reco_z")

	variable_map.add("MCNeutrino","pdg_encoding","pdg_code")
	variable_map.add("MCNeutrino","x","true_x") #TODO Michael gets these in a different way, check this...
	variable_map.add("MCNeutrino","y","true_y")
	variable_map.add("MCNeutrino","z","true_z")

	variable_map.add("I3MCWeightDict","PrimaryNeutrinoEnergy","true_energy")
	variable_map.add("I3MCWeightDict","InteractionType","interaction")
	variable_map.add("I3MCWeightDict","OneWeight","one_weight")
	variable_map.add("I3MCWeightDict","NEvents","n_events")

	#TODO Need to add flux info, where to do this?


	#
	# Define metadata
	#

	metadata = {
		"sample":"greco",
	}


	#
	# Run the conversion
	#

	output_file = "greco_sim.hdf5"

	if dummy_data :
		generate_dummy_hdf5_file(output_file,variable_map,num_events=10)

	else :
		#Call the main i3 -> PISA converter function to do the heavy lifting
		convert_i3_to_pisa(input_data=input_data,
							output_file=output_file,
							variable_map=variable_map,
							metadata=metadata) 



	#
	# Post-processing
	#

	#Re-open the file for editing
	reopened_output_file = tables.open_file(output_file, mode="r+")

	#Combine pegleg track and cascade hypotheses
	combine_reco_track_and_cascade_hypotheses(reopened_output_file)

	#Split neutrino events by nu vs nubar, and CC vs NC
	subdivide_neutrino_groups(reopened_output_file)

	#Calculate effective area (weight)
	calc_effective_area_weight(reopened_output_file)

	#Close the re-opened file
	reopened_output_file.close()



	#
	# Done
	#

	end_time = datetime.datetime.now()

	print ""
	print "GRECO sim i3 -> PISA event file conversion complete :"
	print "  Took : %s" % (end_time-start_time)
	print "  Output file : %s" % (output_file)
	print ""



