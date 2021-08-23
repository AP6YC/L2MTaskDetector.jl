using L2MTaskDetector
using Test
using AdaptiveResonance
using DelimitedFiles

using ConfParser
using Logging

# Set the log level
LogLevel(Logging.Info)

@testset "Config" begin
    # Set the logging level to Info and standardize the random seed
    LogLevel(Logging.Info)

    # Load and parse the configuration
    conf = ConfParse("../data/config/config.ini")
    parse_conf!(conf)

    # Load some parameters and check their contents and types
    mean_file = retrieve(conf, "params", "mean_file")
    @info "Mean file" mean_file typeof(mean_file)
    windows = parse(Int, retrieve(conf, "params", "windows"))
    @info "Windows" windows typeof(windows)
end

@testset "Params" begin

    # Load and parse the configuration
    # conf = ConfParse(projectdir("data","config","config.ini"))
    conf = ConfParse("../data/config/config.ini")
    parse_conf!(conf)

    # Include the parameters file
    # include(projectdir("julia", "taskdetector", "params.jl"))
    include("../src/params.jl")

    # Load the parameters struct
    params = TaskDetectorParameters(conf)

end