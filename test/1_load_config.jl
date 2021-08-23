using ConfParser
using DrWatson
using Logging

# Set the logging level to Info and standardize the random seed
LogLevel(Logging.Info)

# Load and parse the configuration
conf = ConfParse(projectdir("data","config","config.ini"))
parse_conf!(conf)

# Load some parameters and check their contents and types
mean_file = retrieve(conf, "params", "mean_file")
@info "Mean file" mean_file typeof(mean_file)
windows = parse(Int, retrieve(conf, "params", "windows"))
@info "Windows" windows typeof(windows)
