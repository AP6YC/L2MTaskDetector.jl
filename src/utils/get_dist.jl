using Revise
using DelimitedFiles
using Logging
using StatsBase

# Set the log level
LogLevel(Logging.Info)

# Read all the paths as subdirs from this datadrop
# data_dirs = readdir("E:\\dev\\mount\\data\\average_activations", join=true)
# data_dirs = readdir("work/data/average_activations", join=true)
data_dir = "C:\\Users\\sap62\\mount\\combined"
data_dirs = readdir(data_dir, join=true)

# Declare all the cells that we test
# cells = [1, 2, 3, 4]
cells = [1]

# --------------------------------------------------------------------------- #
# Linear example
# --------------------------------------------------------------------------- #
for cell in cells
    # cell = cells[1]
    # Gather the data, training targets, and condensed label list
    data, targets, labels, seq_ind = collect_all_activations_labeled_sequential(data_dirs, cell)
    dim, n_samples = size(data)

    # Get the distribution
    dt = fit(ZScoreTransform, data, dims=2)

    # Save the means
    # mean_file = "julia/experiments/vers_results/mean.csv"
    # scale_file = "julia/experiments/vers_results/scale.csv"
    mean_file = "work/results/preprocess/mean/" * string(cell) * ".csv"
    scale_file = "work/results/preprocess/scale/" * string(cell) * ".csv"
    open(mean_file, "w") do io
        writedlm(io, dt.mean, ",")
    end
    open(scale_file, "w") do io
        writedlm(io, dt.scale, ",")
    end
    # writedlm(mean_file, dt.mean, ",")
    # writedlm(scale_file, dt.scale, ",")

    local_mean = readdlm(mean_file, ',', Float64, '\n')
    local_scale = readdlm(scale_file, ',', Float64, '\n')

    @info local_mean .== dt.mean
    @info local_scale .== dt.scale

    @info vec(local_mean) == dt.mean
    @info vec(local_scale) == dt.scale

    # new_dt = ZScoreTransform(128, 2, vec(local_mean), vec(local_scale))
    new_dt = ZScoreTransform(128*(cell^2), 2, vec(local_mean), vec(local_scale))

    @info size(data)
    x_norm = StatsBase.transform(new_dt, data)
end
