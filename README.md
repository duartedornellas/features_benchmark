# features_benchmark
This repository contains code for the runtime benchmarking of feature detection, description and matching  with OpenCV 3.3.1 implementations. Two output files are generated, one for the feature detectors, and one for the feature descriptors (computed on FAST features). Each line corresponds to an iteration, and is formatted as follows:

'results_detectors.txt':	Shi-Tomasi	SIFT	MSER	FAST	SURF	BRISK	ORB

'results_descriptors.txt': 	SIFT	SURF	BRIEF	BRISK	ORB	FREAK
