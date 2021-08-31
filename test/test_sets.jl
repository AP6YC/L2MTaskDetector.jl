using L2MTaskDetector
using Test
using AdaptiveResonance
using DelimitedFiles

using ConfParser
using Logging
using Random

# Set the log level
LogLevel(Logging.Info)

@testset "Preprocess" begin
    include("../src/preprocess.jl")
    using .YOLOPreprocess

    params = YOLOPreprocessParameters()

    random_3d = rand(28, 28, 128)
    preprocessed = feature_preprocess(random_3d, params)

    params_2 = YOLOPreprocessParameters(
        mean_file="config/mean/2.csv",
        scale_file="config/scale/2.csv",
        windows=2
    )
    preprocessed = feature_preprocess(random_3d, params_2)
end

# Test that the configuration files load correctly
@testset "Config" begin
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
