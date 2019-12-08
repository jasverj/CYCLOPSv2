using DataFrames, Statistics, LinearAlgebra, StatsBase, MultivariateStats, Distributions, CSV, Juno



#------------------#
# Data Processing #
#----------------#



#####################
# begin makefloat! #
#############################################################
# Convert Array{Any} | Array{String} to Array{Float32}     #
###########################################################
# Method 1: Array{Any} | Array{String} to Array{Flaot32} #
#-------------------------------------------------------#
# All strings must be convertible to numbers           #
#-----------------------------------------------------#
function makeFloat!(ar::Array{Any})
    for col in 1:size(ar)[2] # size[2] is the number of columns
        for row in 1:size(ar)[1] # size[1] is the number of rows
            if typeof(ar[row,col]) == String # If the type of a field is string...
                ar[row, col] = parse(Float32, ar[row,col]) # ...that field will be converted to Float32
            end #--end if--#
        end #--end for--#
    end #--end for--#

    ar = convert(Array{Float32, 2}, ar) # Now that all fields are type Float32 the Array{Any} can be converted to Array{Float32}
end
#--end method--#



#---------------------------------------------------------------#
# Method 2/3: Array{Any,1} | Array{String,1} to Array{Float32} #
#-------------------------------------------------------------#
# Will additionally convert from row to column vector        #
#-----------------------------------------------------------#
function makeFloat!(ar::Array{Any,1}, flip::Bool=false)
    for row in 1:size(ar)[1] # size[1] is the number of rows
        if typeof(ar[row]) == String # If the type of a field is string...
            ar[i] = parse(Float32, ar[i]) # ...that field will be converted to Float32
        end #--end if--#
        if flip == true # If flip is true...
            ar = convert(Array{Float32,2}, ar') # ...the array is converted from a row vector to a column vector...
		else # ...otherwise if flip is false...
			ar = convert(Array{Float32,1}, ar) # ...just return Array{Float32,1}
        end #--end if--#
    end #--end for--#

	ar # return ar
end
#--end method--#



#--------------------------------------------#
# Method 4: DataFrame to Array{Float32}     #
#------------------------------------------#
# goes directly from df to Array{Float32} #
#----------------------------------------#
function makeFloat!(df::DataFrame)
    ar = convert(Matrix, df) # Fist convert the DataFrame to a matrix. If strings are contained it will become an Array{Any}
    for col in 1:size(ar)[2] # For each column
        for row in 1:size(ar)[1] # For each row
            if typeof(ar[row,col]) == String # Check if the element is a string...
                ar[row, col] = parse(Float32, ar[row,col]) # ...and convert it to Float32
            end #--end if--#
        end #--end for--#
    end #--end for--#

    ar = convert(Array{Float32, 2}, ar) # Now that all elements are type Float32 the Array{Any} can be converted to an Array{Float32}
end
#--end method--#
###################
# end makefloat! #
#################



#####################
# begin findNAtime #
#################################################################
# Find indices of fields containting string NA and return list #
###############################################################
# Method 1: DataFrame                                        #
#-----------------------------------------------------------#
function findNAtime(df)
    r = [] # Initialize vector for storing indeces
        for ii in 1:length(df) # Go through each element in the DataFrame...
            if typeof(df[ii]) == String # ...check if is type string...
                append!(r, ii) # ...if so, add the index to the list.
            end #--end if--#
        end #--end for--#

    r # return the list containing indeces for elements containing the string NA
end
#--end method--#
###################
# end findNAtime #
#################



######################
# begin getSeedData #
######################################################################################
# Find known genes with minimum average expression and with CV within a given range #
####################################################################################
# Method 1: Array{Float32}                                                        #
#--------------------------------------------------------------------------------#
function getSeedData(OG_data, symbols_of_interest, data_symbol_list, maxCV, minCV, minMean, bluntPercent)
	OG_data = Array{Float32}(OG_data) # Convert Array{Float32} to increase speed
	cleaned_data = removeOutliers!(OG_data, bluntpercent) # Remove outliers according to blunt and replace lowest and highest value with new min and max

	gene_means = vec(mean(cleaned_data, dims=2)) # Get the means of each row
	gene_sds = vec(std(cleaned_data; dims=2)) # Get the standard deviation of each row
	gene_cvs = gene_sds ./ gene_means # Get the coefficient of variation (CV) -- CV is a good measure of relative variability

	criteria1 = findall(in(symbols_of_interest), data_symbol_list) # Find genes list of known genes in full list of genes from data set
	criteria2 = findall(gene_means .> minMean) # Find all genes with a mean expression level greater than the minimum (user defined) mean
	criteria3 = findall(gene_cvs .> minCV) # Find all genes with a CV greater than the minimum (user defined) CV...
	criteria4 = findall(gene_cvs .< maxCV) # ...but smaller than the maximum (user defined) CV.

	allCriteria = intersect(criteria1, criteria2, criteria3, criteria4) # Find all the genes for which all conditions are true
	seed_data = OG_data[allCriteria, :] # Store all samples of genes that meat all the criteria in seed_data
	seed_symbols = data_symbol_list[allCriteria, :] # Store all the names of the genes being kept in seed_symbols

	seed_symbols, seed_data # Return the stored gene names and their respective samples
end
#--end method--#
####################
# end getSeedData #
##################



##########################
# begin removeOutliers! #
#####################################################################################
# Replace outliers from data according to bluntpercent with new min and max values #
###################################################################################
# Method 1: Array{Float32,2}                                                     #
#-------------------------------------------------------------------------------#
# INFO: Max for bluntPercent = 1, reasonable value for bluntPercent = 0.975    #
#-----------------------------------------------------------------------------#
function removeOutliers!(data::Array{Float32, 2}, bluntPercent)
	ngenes, nsamples = size(data) # Number of rows = ngenes, number of columns = nsamples
	nfloor = Int(1 + floor((1 - bluntPercent) * nsamples)) # Index of lowest value to be kept
	nceiling = Int(ceil(bluntPercent*nsamples)) # Index of highest value to be kept
	for row in 1:ngenes # Go through each row
		sorted = sort(vec(data[row, :])) # Sort each row from lowest to highest
		vfloor = sorted[nfloor] # Find the value of the desired new minimum
		vceil = sorted[nceiling] # Find the value of the desired new maximum
		for sample in 1:nsamples # Go through each sample of each row
			data[row, sample] = max(vfloor, data[row, sample]) # If the value of a given field is lower than the new minimum it will be replaced with the new minimum
			data[row,sample] = min(vceil, data[row, sample]) # If the value of a given field is higher than the new maximum it will be replaced with the new maximum
		end #--end for--#
	end #--end for--#

	data
end
#--end method--#
########################
# end removeOutliers! #
######################



########################
# begin getEigengenes #
####################################################################################################################################################################
# Convert data to SVD space, keeping eigengenes that contribute a minimum (user defined) amount of variance, up to minimum (user defined) total variance captured #
##################################################################################################################################################################
# Method 1: Array{Float32,2}                                                                                                                                    #
#--------------------------------------------------------------------------------------------------------------------------------------------------------------#
# INFO: Max for total_var_cap = 1, reasonable value for total_var_cap = 0.97; Max for indiv_var_cont = 1, reasonable value for indiv_var_cont = 0.025 - 0.05  #
#------------------------------------------------------------------------------------------------------------------------------------------------------------#
function getEigengenes(seed_data::Array{Float32, 2}, total_var_cap::Number, indiv_var_cont::Number, maxneigg::Number=30)
    svd_obj = svd(seed_data) # Convert data to SVD object containing singular values and eigengene expression data in SVD space
    expvar = cumsum(svd_obj.S.^2, dims = 1) / sum(svd_obj.S.^2) # .S are the singular values, sorted in descending order. Find the Fraction variance that an eigengene and each eigengene before it makes up from the total variance.

    ReductionDim1 = 1 + length(expvar[expvar .<= total_var_cap]) # How many eigengenes need to be included to have captured the minimum (user specified) variance from the eigengenes (from their singular values)
    vardif = diff(expvar, dims = 1) # Find the difference between each added eigengene
    ReductionDim2 = 1 + length(vardif[vardif .>= indiv_var_cont]) # which eigengenes contribute the minimum (user specified) variance
    ReductionDim = min(ReductionDim1, ReductionDim2, maxneigg) # The last criteria is the maximum number of eigengenes (user specified) that will be kept. Whichever is the smallest is the number of eigengenes kept

    Transform = svd_obj.V[:, 1:ReductionDim]' # .V is the expression of the eigengenes (in SVD space)

    ReductionDim, Array{Float32,2}(10*Transform) # Return the number of eigengenes kept, and the expression of the eigengenes.
end
#--end method--#
######################
# end getEigengenes #
####################



#----------------------#
# end Data Processing #
#--------------------#

#-----------------------------------------------------------------------------#

#--------------------#
# Data Manipulation #
#------------------#



#######################
# begin genSynthData #
#############################################################################
# Generate synthetic data from real data                                   #
###########################################################################
# Method 1: DataFrame to Array{Float32}                                  #
#-----------------------------------------------------------------------#
# Original data is scaled and offset is added to create synthetic data #
#---------------------------------------------------------------------#----------------------------------------------------------------------------------------------------------#
# INFO: this function can only be used if the first and second row contain subject number and time of death, and the first through third column contain probes and gene symbols #
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
function gensynthdata(OGdf::DataFrame, SF::Number, offset::Number=0.0, LOG=false)
    stdev = abs.(SF - 1)/3.8905 # 99.9% lie between 1 and 1+2*(SF-1)
    offsetstd = offset/2.575 # 95% lie between ± offset
    ogdata = CYCLOPS_PrePostProcessModule.makeFloat!(OGdf[3:end,4:end]) # convert data to matrix of Float32
	if LOG == false # if LOG is false use normal distribution for scaling factor
    	synthData = (SF .+ stdev .* randn(size(ogdata,1))) .* ogdata .+ (mean(ogdata, dims = 2) .* (offsetstd .* randn(size(ogdata,1)))) # SF from N~(x̄,σ^2) and (percent of mean) offset to create synth data
	else # or LOG is true and log-normal distribution is used
		synthData = exp(stdev .* randn(size(ogdata,1))) .* ogdata .+ (mean(ogdata, dims = 2) .* (offsetstd .* randn(size(ogdata,1)))) # SF from exp(N~(x̄,σ^2)) and (percent of mean) offset to create synth data
	end #--end if--#
    batchsize = size(ogdata,2)

    batchsize, [ogdata syndata] # new full data set
end
#--end method--#
#####################
# end genSynthData #
###################




end
#--end module--#
