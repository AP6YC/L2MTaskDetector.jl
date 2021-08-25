"""
    preprocess.jl

Description:
    All of the logic for loading and storing auxiliary parameters for the TaskDetector.

Objects:
- TaskDetectorParameters

Authors:
- Sasha Petrenko <sap625@mst.edu> <sap625@umsystem.edu>
"""

# Add shared functions for utilities
include("lib_utils.jl")

# Include distribution definitions
include("get_dist.jl")
