module L2MTaskDetector

# greet() = print("Hello World!")

# Include all files
include("taskdetectormodule.jl")    # TaskDetector logic
include("utils/utils.jl")           # Utilities (compute weights, etc.)

# Export all public names
export
    TaskDetectorModule

end # module
