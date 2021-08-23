"""
    metrics.jl

Description:
    All of the objects and functions for computing C3 performance metrics.

Objects:
- TaskDetectorMetrics

Authors:
- Sasha Petrenko <sap625@mst.edu> <sap625@umsystem.edu>
"""

using StatsBase
using ClusterValidityIndices

"""
    TaskDetectorMetrics

Stateful information of the C3's metrics.

# Fields
- `cvis::Vector{AbstractCVI}`: list of CVIs used in computing metrics for C3.
- `values::Vector{Vector{Float64}}`: most recent CVI criterion values.
- `performance::Float64`: Meta-ICVI value.
- `n_window::Int64`: number of values to track for Meta-ICVI calculation.
"""
mutable struct TaskDetectorMetrics
    cvis::Vector{AbstractCVI}
    values::Vector{Vector{Float64}}
    performance::Float64
    n_window::Integer
end # TaskDetectorMetrics

"""
    TaskDetectorMetrics()

Default constructor for the C3 metrics object.
"""
function TaskDetectorMetrics()
    # Declare all of the cvis used in C3
    cvis = [
        PS(),
        GD43()
    ]

    # Create a container for their criterion values for quick access
    # values = zeros(Float64, length(cvis))
    values = Vector{Vector{Float64}}()
    for i = 1:length(cvis)
        push!(values, Vector{Float64}())
    end

    # Total performance
    performance = 0

    # Default number of windows
    n_window = 5

    # Construct the metrics container
    TaskDetectorMetrics(
        cvis,               # cvis
        values,             # values
        performance,        # performance
        n_window            # n_window
    )
end # TaskDetectorMetrics()

"""
    update_metrics(metrics::TaskDetectorMetrics, sample::Array, label::Int)

Update the task detector's metrics using ICVIs.

# Fields
- `metrics::TaskDetectorMetrics`: the metrics object being updated.
- `sample::Array`: the array of features that are clustered to the label.
- `label::Int`: the label prescribed by the clustering algorithm.
"""
function update_metrics(metrics::TaskDetectorMetrics, sample::Array, label::Integer)
    # Get the number of cvis each time, accomodating changes during operation
    n_cvis = length(metrics.cvis)

    # If the sample was not misclassified and we have a big enough window
    if label != -1
        # Update all of the cvis incrementally
        for ix = 1:n_cvis
            value = get_icvi!(metrics.cvis[ix], sample, label)
            push!(metrics.values[ix], value)
        end
        # If the window is big enough, compute the performance
        if length(metrics.values[1]) >= metrics.n_window
            # Get the spearman correlation
            performance = corspearman(metrics.values[1], metrics.values[2])/2 + 0.5
            # Sanitize a potential NaN response
            metrics.performance = isequal(performance, NaN) ? 0 : performance
        end
    else
        # Default to 0
        metrics.performance = 0
    end

    # FIFO the list
    for ix = 1:n_cvis
        while length(metrics.values[ix]) > metrics.n_window
            popfirst!(metrics.values[ix])
        end
    end

    # Calculate the performance
    # metrics.performance = sigmoid(8*(mean(metrics.values) - 0.5))

    # Return that performance
    return metrics.performance
end # update_metrics(metrics::TaskDetectorMetrics, sample::Array, label::Integer)
