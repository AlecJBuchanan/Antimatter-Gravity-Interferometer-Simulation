README

Prerequisites
	- Wokrstation with a CUDA compatible GPU
	Libraries:
		- numba
		- cython
		- llvmpy
		- build-essentials
		- llvm
		- python-pip
		- NVIDIA toolkit

Installing
	- sudo apt-get install build-essential
	- sudo apt-get install llvm
	- sudo apt-get install python-pip
	- sudo pip install llvmpy
	- sudo pip install cython
	- sudo pip install numba

	Environment Variables:
		- export NUMBAPRO_LIBDEVICE=/usr/local/cuda-9.1/nvvm/libdevice
		- export NUMBAPRO_NVVM=/usr/lib/x86_64-linux-gnu/libnvvm.so
		- export PATH=/usr/local/cuda-9.1/bin:$PATH
Setup
	- From line 241 to line 257 the program parameters are defined.
	#Define initial parameters 
	######################################################
	screen_distance = 5e7 #nm
	screen_length = 1e7
	second_grating_distance = 5e7 #nm
	wavelength = .56 #nm
	U_0 = 1 #?
	wavenumber = 2 * np.pi / wavelength
	numOfSlits = 100 # number of slits in each grating
	numOfPointSources = 100  # number of point sources in each slit
	numObsPoints = numOfSlits * numOfPointSources    # number of observing points on the screen
	spacingType = 'uniform'
	slitLength = 50 #nm
	newSimulation = False
	timings = []
	##########################################################################################
	

Executing
	To run the program, go to the terminal and 
		cd into the directory where 2GratingDiffreaction_final.py is stored.

	Run the program with: 
		$ python 2GratingDiffreaction_final.py 

Built with
	- numpy
	- numba
	- math
	- cmath
	- matplot
	- sys
	- time

Authors
	Alec Buchanan
	Patrick Connolly

Acknowledgements
	jarroyo3@hawk.iit.edu
	blowell@hawk.iit.edu
	Dr. Derrick Mancini	

