using DrWatson
using ConfParser

# Load and parse the configuration
conf = ConfParse(projectdir("data","config","config.ini"))
parse_conf!(conf)

# Include the parameters file
include(projectdir("julia", "taskdetector", "params.jl"))

# Load the parameters struct
params = TaskDetectorParameters(conf)
