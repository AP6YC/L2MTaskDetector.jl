using L2MTaskDetector
using Documenter

DocMeta.setdocmeta!(L2MTaskDetector, :DocTestSetup, :(using L2MTaskDetector); recursive=true)

makedocs(;
    modules=[L2MTaskDetector],
    authors="Sasha Petrenko",
    repo="https://github.com/AP6YC/L2MTaskDetector.jl/blob/{commit}{path}#{line}",
    sitename="L2MTaskDetector.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://AP6YC.github.io/L2MTaskDetector.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/AP6YC/L2MTaskDetector.jl",
)
