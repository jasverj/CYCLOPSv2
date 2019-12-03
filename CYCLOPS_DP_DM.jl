using DataFrames, Statistics, LinearAlgebra, StatsBase, MultivariateStats, Distributions, CSV, Juno

#########################################################
# Convert Array{Any} | Array{String} to Array{Float32} #
#######################################################
# Array{Any} | Array{String} to Array{Flaot32}  #
################################################
# All strings must be convertible to numbers  #
function makefloat!(ar::Array{Any}) # convert to Array{Any} first, using convert{Matrix, df}.
    for col in 1:size(ar)[2]
        for row in 1:size(ar)[1]
            if typeof(ar[row,col]) == String
                ar[row, col] = parse(Float32, ar[row,col])
            end
        end
    end
    ar = convert(Array{Float32, 2}, ar)
end

# Array{Any,1} | Array{String,1} to Array{Float32}      #
########################################################
# Will additionally convert from row to column vector #
function makefloat!(ar::Array{Any,1}, conv::Bool=false)
    for col in 1:size(ar)[1]
        if typeof(ar[col]) == String
            ar[i] = parse(Float32, ar[i])
        end
        if conv == true
            convert(Array{Float32,2}, ar')
        end
    end
end

# DataFrame to Array{Float32}               #
############################################
# goes directly from df to Array{Float32} #
function makefloat!(df::DataFrame) # will convert to Array{Float} first
    ar = convert(Matrix, df)
    for col in 1:size(ar)[2]
        for row in 1:size(ar)[1]
            if typeof(ar[row,col]) == String
                ar[row, col] = parse(Float32, ar[row,col])
            end
        end
    end
    ar = convert(Array{Float32, 2}, ar)
end
