using AdaptiveResonance

include("params.jl")
include("preprocess.jl")
include("metrics.jl")

"""
    TaskDetectorModule

The stateful information of C3.

# Fields
- `ddvfa::DDVFA`: the first processing step of C3.
- `metrics::TaskDetectorMetrics`: metrics container for C3, containing ICVIs and more.
- `task_map::Dict{String, Int}`: internal mapping of string task names to incremental indices.
- `task::L2MTask`: the most recent task belief of C3.
- `params::TaskDetectorParameters`: all algorithmic parameters.
"""
mutable struct TaskDetectorModule
    # Logic of the module
    ddvfa::DDVFA            # C3: Step 1

    # Metrics of the module
    metrics::TaskDetectorMetrics

    # Processing parameters
    params::TaskDetectorParameters

    # Statistics for diagnosis
    stats::Dict
end # TaskDetectorModule

"""
    TaskDetectorModule(conf::ConfParse)

Default constructor for the C3 stateful information, requires an L2M config.

# Arguments
- `conf::ConfParse`: the L2M config object for the task detector.
"""
function TaskDetectorModule(conf::ConfParse)
    # Get the algorithmic parameters
    params = TaskDetectorParameters(conf)

    # Instantiate the DDVFA submodule
    ddvfa = DDVFA(params.opts_ddvfa)

    # Instantiate the C3 metrics container
    metrics = TaskDetectorMetrics()

    # Manually set the DDVFA data configuration
    dim = params.feat_dim * params.windows    # ART dimension is features * windows
    feat_min = 0.0                          # Sigmoid min = 0.0
    feat_max = 1.0                          # Sigmoid max = 1.0
    data_config = DataConfig(
        true,
        repeat([feat_min], dim),    # min
        repeat([feat_max], dim),    # max
        dim,                        # dim
        dim * 2                       # dim_comp
    )
    ddvfa.config = data_config

    # Set up the statistics dictionary with default values
    stats = Dict(
        "n_categories"  => 0,
        "n_weights"     => 0
    )

    # Create the module kernel
    TaskDetectorModule(
        ddvfa,                  # ddvfa
        metrics,                # metrics
        params,                 # params
        stats                   # stats
    )
end # TaskDetectorModule()

"""
    train!(taskmod::TaskDetectorModule, x::Array{T, 1} ; y=0)  where {T<:Real}

Train the C3 art module in place on an array.

# Arguments
- `taskmod::TaskDetectorModule`: the task detector algorithmic module.
- `x::Array{Real, 1}`: the input sample (assumes preprocessed and 1-D).
- `y=0`: incremental integer supervisory label, default to empty for unsupervised.
"""
function train!(taskmod::TaskDetectorModule, x::Array{T,1} ; y=0)  where {T <: Real}
    # Convert provided label into correct format for train!
    label = y == 0 ? [] : [y]

    # Train and categorize the DDVFA layer on the data
    y_hat = AdaptiveResonance.train!(taskmod.ddvfa, x, y=label)

    # Get the DDVFA stats, total number of categories, etc.
    n_categories = taskmod.ddvfa.n_categories
    taskmod.stats["n_categories"] = n_categories
    taskmod.stats["n_weights"] = sum([taskmod.ddvfa.F2[i].n_categories for i = 1:n_categories])

    # Return the trained class during supervised and unsupervised training
    return y_hat
end # train!(taskmod::TaskDetectorModule, x::Array{T, 1} ; y=0)  where {T<:Real}

"""
    classify(taskmod::TaskDetectorModule, x::Array{T, 1}) where {T<:Real}

Get the task from the features.

# Arguments
- `taskmod::TaskDetectorModule`: the task detector algorithmic module.
- `x::Array{Real, 1}`: the input sample (assumes preprocessed and 1-D).
"""
function classify(taskmod::TaskDetectorModule, x::Array{T,1}) where {T <: Real}
    # Get the DDVFA categories from the data, using bmu if mismatched
    y_hat = AdaptiveResonance.classify(taskmod.ddvfa, x, get_bmu=true)

    # Return the task belief without having learned
    return y_hat
end # classify(taskmod::TaskDetectorModule, x::Array{T, 1}, timestamp::UInt64) where {T<:Real}

