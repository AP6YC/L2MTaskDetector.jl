using L2MTaskDetector
using Test
using AdaptiveResonance
using DelimitedFiles

using ConfParser
using Logging

# Set the log level
LogLevel(Logging.Info)

# Test that the configuration files load correctly
@testset "Config" begin
    # Set the logging level to Info and standardize the random seed
    LogLevel(Logging.Info)

    # Load and parse the configuration
    conf = ConfParse("../src/config/config.ini")
    parse_conf!(conf)

    # Load some parameters and check their contents and types
    mean_file = retrieve(conf, "params", "mean_file")
    @info "Mean file" mean_file typeof(mean_file)
    @test typeof(mean_file) == String

    windows = parse(Int, retrieve(conf, "params", "windows"))
    @info "Windows" windows typeof(windows)
    @test typeof(windows) <: Int
end

# Test that the parameter struct loads correctly
@testset "Params" begin
    # Load and parse the configuration
    conf = ConfParse("../src/config/config.ini")
    parse_conf!(conf)

    # Include the parameters file
    include("../src/params.jl")

    # Load the parameters struct
    params = TaskDetectorParameters(conf)

    # @info "Params" params
end

# Test that the module loads correctly
@testset "Module" begin
    # Load and parse the configuration
    conf = ConfParse("../src/config/config.ini")
    parse_conf!(conf)

    tdm = TaskDetectorModule(conf)
end
