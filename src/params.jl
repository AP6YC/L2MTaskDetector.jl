"""
    params.jl

Description:
    All of the logic for loading and storing auxiliary parameters for the TaskDetector.

Objects:
- TaskDetectorParameters

Authors:
- Sasha Petrenko <sap625@mst.edu> <sap625@umsystem.edu>
"""

using StatsBase
using ConfParser
using AdaptiveResonance
using DelimitedFiles
using DrWatson

function get_param(config::ConfParse, block::String, key::String)
    # haskey(s::ConfParse, block::String, key::String)
    if haskey(config, block, key)
        return retrieve(config, block, key)
    else
        error("TaskDetector is trying to load nonexistent parameter: $block - $key")
    end
end

"""
    TaskDetectorParameters

Struct containing the options for the task detector.

# Fields
- `mean::Array{Float64, 1}`: means of each feature, used by transformer.
- `scale::Array{Float64, 1}`: scaling parameter of each feature, used by transformer.
- `transformer::ZScoreTransform`: object for transforming the data during preprocessing.
- `windows::Int`: number of cells in each dimension (i.e., windows=2 results in 4 panels).
- `feat_dim::Int`: dimension of each individual feature, before windowing.
- `opts_ddvfa::opts_DDVFA`: the DDVFA options object.
"""
mutable struct TaskDetectorParameters
    # Affine transformation parameters
    mean::Array{Float64,1}
    scale::Array{Float64,1}
    transformer::ZScoreTransform

    # Number of windows
    # NOTE: this is actually number of cells in each dimension
    #   e.g. windows = 2 results in four panels in the 2D image
    windows::Integer

    # Dimension of an individual feature, before considering windowing
    feat_dim::Integer

    # Sigmoid scaling parameter
    sigmoid_scaling::Float64

    # ART module options
    opts_ddvfa::opts_DDVFA

end # TaskDetectorParameters


"""
    TaskDetectorParameters(config::ConfParse)

Default intialization for the task detector parameters.

# Arguments
- `config::ConfParse`: global configuration object with fields for preprocessing.
"""
function TaskDetectorParameters(config::ConfParse)

    # Get the means and scaling parameters
    mean_file = get_param(config, "params", "mean_file")
    scale_file = get_param(config, "params", "scale_file")
    # Get the number of windows
    windows = parse(Int, get_param(config, "params", "windows"))
    # Get the feature dimension
    feat_dim = parse(Int, get_param(config, "params", "feat_dim"))
    # Get the sigmoid scaling parameter
    sigmoid_scaling = parse(Float64, get_param(config, "params", "sigmoid_scaling"))

    # Load the mean and scale local configs
    mean = vec(readdlm(projectdir(mean_file), ',', Float64, '\n'))
    scale = vec(readdlm(projectdir(scale_file), ',', Float64, '\n'))

    # Construct the transformer from the mean and scale parameters
    transformer = ZScoreTransform(windows * feat_dim, 2, mean, scale)

    # Construct the DDVFA options from the taskdet_config.ini with defaults
    opts_ddvfa = opts_DDVFA(
        rho_lb=parse(Float64, get_param(config, "ddvfa", "rho_lb")),
        rho_ub=parse(Float64, get_param(config, "ddvfa", "rho_ub")),
        alpha=parse(Float64, get_param(config, "ddvfa", "alpha")),
        beta=parse(Float64, get_param(config, "ddvfa", "beta")),
        gamma=parse(Float64, get_param(config, "ddvfa", "gamma")),
        gamma_ref=parse(Float64, get_param(config, "ddvfa", "gamma_ref")),
        method=get_param(config, "ddvfa", "method"),
        display=parse(Bool, get_param(config, "ddvfa", "display"))
    )

    # Construct and return the parameters
    TaskDetectorParameters(
        mean,
        scale,
        transformer,
        windows,
        feat_dim,
        sigmoid_scaling,
        opts_ddvfa,
    )
end # TaskDetectorParameters(config::ConfParse)
