using AdaptiveResonance

using DelimitedFiles
using Logging
using Random
using StatsBase
using Statistics
# Post-processing functions

"""
    collect_activations(data_dir::String)

Return the activations from a single directory
"""
function collect_activations(data_dir::String)
    data_full = []
    files = readdir(data_dir)
    for i_file in 1:1:length(files)
        local_data_raw = readdlm(joinpath(data_dir, files[i_file]), ',')
        # Permute the data to column-major format
        local_data = permutedims(local_data_raw)
        # Reshape the array to a single column if there are multiple
        if size(local_data)[2] > 1
            local_data = reshape(local_data, :, 1)
        end
        if isempty(data_full)
            data_full = Array{Float64}(undef, size(local_data)[1], 1)
            data_full[:, 1] = local_data
        else
            data_full = hcat(data_full, local_data)
        end
    end
    return data_full
end

"""
    collect_all_activations(data_dirs::Array, cell::Int)

Return just the yolo activations from a list of data directories.
"""
function collect_all_activations(data_dirs::Array, cell::Int)
    data_grand = []
    for data_dir in data_dirs
        data_dir_full = joinpath(data_dir, string(cell))
        data_full = collect_activations(data_dir_full)
        # If the full data struct is empty, initialize with the size of the data
        if isempty(data_grand)
            data_grand = Array{Float64}(undef, size(data_full)[1], 1)
        end
        data_grand = [data_grand data_full]
    end
    return data_grand
end

"""
    collect_all_activations_labeled_sequential(data_dirs::Array, cell::Int)

Return the yolo activations, training targets, and condensed labels list from a list of data directories along with the category indices.
"""
function collect_all_activations_labeled_sequential(data_dirs::Array, cell::Int)
    data_grand = []
    labels = []
    targets = []
    seq_ind = []
    # for data_dir in data_dirs
    for i = 1:length(data_dirs)
        # Get the full local data directory
        data_dir = data_dirs[i]
        data_dir_full = joinpath(data_dir, string(cell))

        # Assign the directory as the label
        push!(labels, basename(data_dir))

        # Get all of the data from the full data directory
        data_full = collect_activations(data_dir_full)
        dim, n_samples = size(data_full)

        # If the full data struct is empty, initialize with the size of the data
        if isempty(data_grand)
            # data_grand = Array{Float64}(undef, size(data_full)[1], 1)
            data_grand = Array{Float64}(undef, dim, 0)
        end

        # Set the labeled targets
        # targets = vcat(targets, repeat([i], size(data_full)[2]))
        for j = 1:n_samples
            push!(targets, i)
        end

        # Set the "ranges" of the indices
        if i == 1
            push!(seq_ind, [1, n_samples])
        else
            start_ind = seq_ind[i-1][2] + 1
            push!(seq_ind, [start_ind, start_ind + n_samples - 1])
        end
        # Concatenate the most recent batch with the grand dataset
        data_grand = [data_grand data_full]
    end
    return data_grand, targets, labels, seq_ind
end

"""
    collect_all_activations_labeled(data_dirs::Array, cell::Int)

Return the yolo activations, training targets, and condensed labels list from a list of data directories.
"""
function collect_all_activations_labeled(data_dirs::Array, cell::Int)
    data_grand = []
    labels = []
    targets = []
    # for data_dir in data_dirs
    for i = 1:length(data_dirs)
        # Get the full local data directory
        data_dir = data_dirs[i]
        data_dir_full = joinpath(data_dir, string(cell))

        # Assign the directory as the label
        push!(labels, basename(data_dir))

        # Get all of the data from the full data directory
        data_full = collect_activations(data_dir_full)
        dim, n_samples = size(data_full)

        # If the full data struct is empty, initialize with the size of the data
        if isempty(data_grand)
            # data_grand = Array{Float64}(undef, size(data_full)[1], 1)
            data_grand = Array{Float64}(undef, dim, 0)
        end

        # Set the labeled targets
        # targets = vcat(targets, repeat([i], size(data_full)[2]))
        for j = 1:n_samples
            push!(targets, i)
        end

        # Concatenate the most recent batch with the grand dataset
        data_grand = [data_grand data_full]
    end
    return data_grand, targets, labels
end

"""
    DataSplit

A basic struct for encapsulating the four components of supervised training.
"""
mutable struct DataSplit
    train_x::Array
    test_x::Array
    train_y::Array
    test_y::Array
    DataSplit(train_x, test_x, train_y, test_y) = new(train_x, test_x, train_y, test_y)
end

"""
    DataSplit(data_x::Array, data_y::Array, ratio::Float)

Return a DataSplit struct that is split by the ratio (e.g. 0.8).
"""
function DataSplit(data_x::Array, data_y::Array, ratio::Real)
    dim, n_data = size(data_x)
    split_ind = Int(floor(n_data*ratio))

    train_x = data_x[:, 1:split_ind]
    test_x = data_x[:, split_ind+1:end]
    train_y = data_y[1:split_ind]
    test_y = data_y[split_ind+1:end]

    return DataSplit(train_x, test_x, train_y, test_y)
end

"""
    DataSplit(data_x::Array, data_y::Array, ratio::Real, seq_ind::Array)

Sequential loading and ratio split of the data.
"""
function DataSplit(data_x::Array, data_y::Array, ratio::Real, seq_ind::Array)
    dim, n_data = size(data_x)
    n_splits = length(seq_ind)

    train_x = Array{Float64}(undef, dim, 0)
    train_y = Array{Float64}(undef, 0)
    test_x = Array{Float64}(undef, dim, 0)
    test_y = Array{Float64}(undef, 0)

    # Iterate over all splits
    for ind in seq_ind
        local_x = data_x[:, ind[1]:ind[2]]
        local_y = data_y[ind[1]:ind[2]]
        # n_data = ind[2] - ind[1] + 1
        n_data = size(local_x)[2]
        split_ind = Int(floor(n_data*ratio))

        train_x = [train_x local_x[:, 1:split_ind]]
        test_x = [test_x local_x[:, split_ind+1:end]]
        train_y = [train_y; local_y[1:split_ind]]
        test_y = [test_y; local_y[split_ind+1:end]]
    end
    return DataSplit(train_x, test_x, train_y, test_y)
end # DataSplit(data_x::Array, data_y::Array, ratio::Real, seq_ind::Array)

"""
    yolo_normalize(x::Real)

Normalize the data from approximately [-0.2, 1.8] to the range [0, 1].

This function is a linear transformation and does not truly limit the range of x.
"""
function yolo_normalize(x::Real)
    # return (x - 0.2) / 2.0
    # return (x - 0.8)
    return (x + 0.2) / 2.0
end #

"""
    sigmoid(x::Real)

Return the sigmoid function on x.
"""
function sigmoid(x::Real)
    # return 1.0 / (1.0 + exp(-x))
    return one(x) / (one(x) + exp(-x))
end

"""
    feature_preprocess!(data_split::DataSplit)
"""
function feature_preprocess!(data_split::DataSplit)
    # Standardize
    dt_train = fit(ZScoreTransform, data_split.train_x, dims=2)
    dt_test = fit(ZScoreTransform, data_split.test_x, dims=2)
    data_split.train_x = StatsBase.transform(dt_train, data_split.train_x)
    data_split.test_x = StatsBase.transform(dt_test, data_split.test_x)

    # Normalize the data according to typical yolo activation ranges (-0.2 - 0.8)
    # data_split.train_x = yolo_normalize.(data_split.train_x)
    # data_split.test_x = yolo_normalize.(data_split.test_x)

    # Squash the data sigmoidally in case of outliers
    data_split.train_x = sigmoid.(data_split.train_x)
    data_split.test_x = sigmoid.(data_split.test_x)
end

function load_sim_data(data_dirs, cell, use_alt)

    split_ratio = 0.8

    # Gather the data, training targets, and condensed label list
    data, targets, labels, seq_ind = collect_all_activations_labeled_sequential(data_dirs, cell)

    # Create a training split
    data_split = DataSplit(data, targets, split_ratio, seq_ind)

    # Standardize
    feature_preprocess!(data_split)

    # If using the altitude, manipulate the final features
    if use_alt
        alt_map = Dict(
            1 => 0,
            2 => 0.5,
            3 => 1
        )
        alt_train = [alt_map[x] for x in data_split.train_y]
        alt_test = [alt_map[x] for x in data_split.test_y]
        data_split.train_x = [data_split.train_x; alt_train']
        data_split.test_x = [data_split.test_x; alt_test']
    end

    return data_split
end

# --------------------------------------------------------------------------- #
# Linear example
# --------------------------------------------------------------------------- #
function C3Sim(d::Dict{String, Any}, data_split::DataSplit)

    # Set the DDVFA options
    ddvfa_opts = opts_DDVFA()
    ddvfa_opts.method = d["method"]
    ddvfa_opts.gamma = d["gamma"]
    ddvfa_opts.rho_ub = d["rho_ub"]
    ddvfa_opts.rho_lb = d["rho_lb"]
    # ddvfa_opts.rho = d["rho"]
    ddvfa_opts.rho = d["rho_lb"]
    ddvfa_opts.display = false

    # Create the ART modules
    art_ddvfa = DDVFA(ddvfa_opts)

    # Get the data stats
    dim, n_samples = size(data_split.train_x)

    # Set the DDVFA config Manually
    art_ddvfa.config = DataConfig(0.0, 1.0, dim)
    @info "dim: $dim"

    # Train the DDVFA model and time it
    train_stats = @timed train!(art_ddvfa, data_split.train_x, y=data_split.train_y)
    y_hat_train = train_stats.value

    # Training performance
    local_train_y = convert(Array{Int}, data_split.train_y)
    train_perf = NaN
    try
        train_perf = performance(y_hat_train, local_train_y)
    catch
        @info "Performance error!"
    end
    @info "Training Performance: $(train_perf)"

    # Testing performance, timed
    test_stats = @timed classify(art_ddvfa, data_split.test_x)
    y_hat_test = test_stats.value
    local_test_y = convert(Array{Int}, data_split.test_y)
    test_perf = NaN
    try
        test_perf = performance(y_hat_test, local_test_y)
    catch
        @info "Performance error!"
    end
    @info "Testing Performance: $(test_perf)"

    # Get the number of weights and categories
    total_vec = [art_ddvfa.F2[i].n_categories for i = 1:art_ddvfa.n_categories]
    total_cat = sum(total_vec)
    # @info "Categories: $(art_ddvfa.n_categories)"
    # @info "Weights: $(total_cat)"

    # Store all of the results of interest
    fulld = copy(d)
    # Performances
    fulld["p_tr"] = train_perf
    fulld["p_te"] = test_perf
    # ART statistics
    fulld["n_cat"] = art_ddvfa.n_categories
    fulld["n_wt"] = total_cat
    fulld["m_wt"] = mean(total_vec)
    fulld["s_wt"] = std(total_vec)
    # Timing statistics
    fulld["t_tr"] = train_stats.time
    fulld["gc_tr"] = train_stats.gctime
    fulld["b_tr"] = train_stats.bytes
    fulld["t_te"] = test_stats.time
    fulld["gc_te"] = test_stats.gctime
    fulld["b_te"] = test_stats.bytes
    # Return the results
    return fulld
end

"""
    load_iris(data_path::String ; split_ratio::Real = 0.8)

Loads the iris dataset for testing and examples.
"""
function load_iris(data_path::String ; split_ratio::Real = 0.8)
    raw_data = readdlm(data_path,',')
    labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    raw_x = raw_data[2:end, 2:5]
    raw_y_labels = raw_data[2:end, 6]
    raw_y = Array{Int64}(undef, 0)
    for ix = 1:length(raw_y_labels)
        for jx = 1:length(labels)
            if raw_y_labels[ix] == labels[jx]
                push!(raw_y, jx)
            end
        end
    end
    n_samples, n_features = size(raw_x)

    # Julia is column-major, so use columns for features
    raw_x = permutedims(raw_x)

    # Shuffle the data and targets
    ind_shuffle = Random.randperm(n_samples)
    x = raw_x[:, ind_shuffle]
    y = raw_y[ind_shuffle]

    data = DataSplit(x, y, split_ratio)

    return data
end # load_iris(data_path::String ; split_ratio::Real = 0.8)

"""
    get_cvi_data(data_file::String)

Get the cvi data specified by the data_file path.
"""
function get_cvi_data(data_file::String)
    # Parse the data
    data = readdlm(data_file, ',')
    data = permutedims(data)
    train_x = convert(Matrix{Real}, data[1:2, :])
    train_y = convert(Vector{Int64}, data[3, :])

    return train_x, train_y
end # get_cvi_data(data_file::String)
