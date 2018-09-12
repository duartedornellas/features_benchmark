# features_benchmark
This repository contains code for the runtime benchmarking of feature detection, description and matching  with OpenCV 3.3.1 implementations. Three output files are generated, 'results_detectors.txt', 'results_descriptors.txt' and 'results_matchers.txt'. For the description and matching tests, the descriptors were computed on FAST features. On these files, each line corresponds to an iteration, and is formatted as follows:

'results_detectors.txt':	Shi-Tomasi	SIFT	MSER	FAST	SURF	BRISK	ORB

'results_descriptors.txt': 	SIFT	SURF	BRIEF	BRISK	ORB   FREAK

'results_matchers.txt': 	SIFT	SURF	BRIEF	BRISK	ORB   FREAK
