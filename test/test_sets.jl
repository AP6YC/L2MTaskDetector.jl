using L2MTaskDetector
using Test
using AdaptiveResonance
using Logging
using DelimitedFiles

# Set the log level
LogLevel(Logging.Info)

@testset "common.jl" begin
    @info "------- Common Code Tests -------"
    # Example arrays
    three_by_two = [1 2; 3 4; 5 6]

    # Test DataConfig constructors
    dc1 = DataConfig()                  # Default constructor
    dc2 = DataConfig(0, 1, 2)           # When min and max are same across all features
    dc3 = DataConfig([0, 1], [2, 3])    # When min and max differ across features

    # Test get_n_samples
    @test get_n_samples([1,2,3]) == 1           # 1-D array case
    @test get_n_samples(three_by_two) == 2      # 2-D array case

    # Test breaking situations
    @test_throws ErrorException performance([1,2],[1,2,3])
    @test_logs (:warn,) AdaptiveResonance.data_setup!(dc3, three_by_two)
    bad_config =  DataConfig(1, 0, 3)
    @test_throws ErrorException linear_normalization(three_by_two, config=bad_config)
end # @testset "common.jl"

@testset "Config" begin

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

end