"""
    preprocess.jl

Description:
    Defines module `YOLOPreprocess`.

    Preprocessing toolchain for the YOLOv3 features coming into C3.

Authors:
    - Sasha Petrenko <sap625@mst.edu> <sap625@umsystem.edu>
"""
module YOLOPreprocess

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using Statistics: mean
using StatsBase
using DelimitedFiles

# -----------------------------------------------------------------------------
# ALIASES
# -----------------------------------------------------------------------------

# Real-numbered aliases
const RealArray{T<:Real, N} = AbstractArray{T, N}
const RealVector{T<:Real} = AbstractArray{T, 1}
const RealMatrix{T<:Real} = AbstractArray{T, 2}
const Real3D{T<:Real} = AbstractArray{T, 3}

# Integered aliases
const IntegerArray{T<:Integer, N} = AbstractArray{T, N}
const IntegerVector{T<:Integer} = AbstractArray{T, 1}
const IntegerMatrix{T<:Integer} = AbstractArray{T, 2}
const Integer3D{T<:Integer} = AbstractArray{T, 3}

# Specifically floating-point aliases
const RealFP = Union{Float32, Float64}

# -----------------------------------------------------------------------------
# MODULES
# -----------------------------------------------------------------------------

"""
    YOLOPreprocessParameters

# Fields
- `transformer::ZScoreTransform`: StatsBase transformer using means and scaling parameters.
- `windows::Int`: Number of window divisions to average features in (e.g. windows=2 for 4 cells, windows=3 for 9, etc.).
- `sigmoid_scaling::Float64`: sigmoid scaling parameter
"""
mutable struct YOLOPreprocessParameters
    transformer::ZScoreTransform
    windows::Int
    sigmoid_scaling::RealFP
end

"""
    YOLOPreprocessParameters(mean_file::String, scale_file::String, windows::Int, feat_dim::Int, sigmoid_scaling::Float64)

# Arguments
- `mean_file::String`: relative location of .csv containing the feature means. Default "config/mean/1.csv".
- `scale_file::String`: relative location of .csv containing the feature scaling parameters. Default "config/scale/1.csv".
- `windows::Int`: number of divisions for feature "windows" (e.g. windows=2 for 4 cells, windows=3 for 9, etc.). Should correspond to mean and scale files. Default 1.
- `feat_dim::Int`: dimensionality of the YOLO features (i.e., length of squashed feature vector). Multiplied by "windows" to get actual post-process dim. Default 128.
- `sigmoid_scaling::Float64`: sigmoid scaling parameter. Default 3.0.
"""
function YOLOPreprocessParameters(;
    mean_file::String="config/mean/1.csv",
    scale_file::String="config/scale/1.csv",
    windows::Int=1,
    feat_dim::Int=128,
    sigmoid_scaling::RealFP=3.0
)
    # Load the mean and scale local configs
    local_path = @__DIR__
    mean = vec(readdlm(joinpath(local_path, mean_file), ',', Float64, '\n'))
    scale = vec(readdlm(joinpath(local_path, scale_file), ',', Float64, '\n'))

    # Construct the transformer from the mean and scale parameters
    transformer = ZScoreTransform(windows^2 * feat_dim, 2, mean, scale)

    # Construct and return the parameters
    YOLOPreprocessParameters(
        transformer,
        windows,
        sigmoid_scaling
    )
end # YOLOPreprocessParameters(mean_file::String, scale_file::String, windows::Int, feat_dim::Int, sigmoid_scaling::Float64)

# -----------------------------------------------------------------------------
# METHODS
# -----------------------------------------------------------------------------

"""
    window_average(data::Real3D, bins::Int)

Average the 3-D data into bins.

For example, if bins = 2, then the features are averaged into two bins per dimension (i.e. four quadrants in a 2-D image), resulting in a vector of length four times the feature length.

# Arguments
- `data::Array`: the input features from C1.
- `bins::Int`: the number of cells in each dimension (see above).
"""
function window_average(data::Real3D, bins::Int)

    # If we are averaging multiple windows
    if bins > 1
        # Explicitly extract the shape of the data
        dim_y, dim_x, dim_feat = size(data)

        # Get the number of pixels for each dimension in each cell
        # TODO: this loses a pixel row for odd dims, make a special case for this
        split_x = Int(floor(dim_x/bins))
        split_y = Int(floor(dim_y/bins))

        # Initialize the return container (always faster to preallocate)
        average_output = zeros(Float64, dim_feat*bins^2)

        # Iterate across all of the cells/bins (indexing at 0 for later ease)
        for iy = 0:bins - 1
            for ix = 0:bins - 1
                # Get the sub array, careful of Julia's 1-indexing
                sub_array = data[
                    split_y*iy + 1 : split_y * (iy + 1),
                    split_x*ix + 1 : split_x * (ix + 1),:]
                # Compute the sub array's average features
                local_averages = mean(sub_array, dims=(1, 2))
                # Assign the averages into the output container
                feat_index = dim_feat*(iy*bins+ix) + 1
                average_output[feat_index : feat_index + dim_feat - 1] = local_averages
            end
        end
    # Otherwise, take the average along x and y all at once
    else
        average_output = vec(mean(data, dims=(1, 2)))
    end

    return average_output
end # window_average(data::Real3D, bins::Int)

"""
    sigmoid(x::RealFP)

Return the sigmoid function on x.

# Arguments
- `x::RealFP`: the value to process through the sigmoid function.
"""
function sigmoid(x::RealFP)
    return one(x) / (one(x) + exp(-x))
end # sigmoid(x::RealFP)

"""
    sigmoid(x::RealArray)

Broadcasts sigmoid(x::RealArray) across an array x.

# Arguments
- `x::RealArray`: array to process sigmoidally.
"""
function sigmoid(x::RealArray)
    return sigmoid.(x)
end # sigmoid(x::RealArray)

"""
    feature_preprocess(x::Real3D, params::YOLOPreprocessParameters)

Preprocess the incoming YOLO features.

# Arguments
- `x::Real3D`: array to preprocess for C3. Three-dimensional.
- `params::YOLOPreprocessParameters`: the algorithm's internal parameters.

# Preprocessing Steps
1. 3-D YOLO features are averaged into windows and concatenated into a single vector.
2. This vector is processed with a Z-score transform by the loaded means and scales in params.
3. This scaled vector is normalized sigmoidally within [0, 1] for each feature.
"""
function feature_preprocess(x::Real3D, params::YOLOPreprocessParameters)
    # Average the data into windows
    x_norm = window_average(x, params.windows)

    # Standardize with the data transformer
    x_norm = StatsBase.transform(params.transformer, x_norm)

    # Sigmoidally squash the data to normalize between [0, 1] scaled by sigmoid_scaling
    x_norm = sigmoid(params.sigmoid_scaling.*(x_norm.*2 .- 1))

    return x_norm
end # feature_preprocess(x::Real3D, params::YOLOPreprocessParameters)

# -----------------------------------------------------------------------------
# EXPORTS
# -----------------------------------------------------------------------------

# Export relevant structures and methods
export

    # Structures
    YOLOPreprocessParameters,

    # Methods
    feature_preprocess

end
