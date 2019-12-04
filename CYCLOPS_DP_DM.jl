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
function makefloat!(ar::Array{Any}) # convert to Array{Any} first, using convert{Matrix, df}.
    for col in 1:size(ar)[2] # size[2] is the number of columns
        for row in 1:size(ar)[1] # size[1] is the number of rows
            if typeof(ar[row,col]) == String # if the type of a field is string...
                ar[row, col] = parse(Float32, ar[row,col]) # ...that field will be converted to Float32
            end #--end if--#
        end #--end for--#
    end #--end for--#

    ar = convert(Array{Float32, 2}, ar) # Now that all fields are type Float32 the Array{Any} can be converted to Array{Float32}
end
#--end function--#

#---------------------------------------------------------------#
# Method 2/3: Array{Any,1} | Array{String,1} to Array{Float32} #
#-------------------------------------------------------------#
# Will additionally convert from row to column vector        #
#-----------------------------------------------------------#
function makefloat!(ar::Array{Any,1}, flip::Bool=false)
    for row in 1:size(ar)[1] # size[1] is the number of rows
        if typeof(ar[row]) == String # if the type of a field is string...
            ar[i] = parse(Float32, ar[i]) # ...that field will be converted to Float32
        end #--end if--#
        if flip == true # if flip is true...
            ar = convert(Array{Float32,2}, ar') # ...the array is converted from a row vector to a column vector...
		else # ...otherwise if flip is false...
			ar = convert(Array{Float32,1}, ar) # ...just return Array{Float32,1}
        end #--end if--#
    end #--end for--#

	ar # return ar
end
#--end function--#

#--------------------------------------------#
# Method 4: DataFrame to Array{Float32}     #
#------------------------------------------#
# goes directly from df to Array{Float32} #
#----------------------------------------#
function makefloat!(df::DataFrame) # will convert to Array{Float} first
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
#--end function--#
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
    r = [] # initialize vector for storing indeces
        for ii in 1:length(df) # go through each element in the DataFrame...
            if typeof(df[ii]) == String # ...check if is type string...
                append!(r, ii) # ...if so, add the index to the list.
            end #--end if--#
        end #--end for--#

    r # return the list containing indeces for elements containing the string NA
end
#--end function--#
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
	OG_data = Array{Float32}(OG_data) # convert Array{Float32} to increase speed
	cleaned_data = removeOutliers!(OG_data, bluntpercent) # remove outliers according to blunt and replace lowest and highest value with new min and max

	gene_means = vec(mean(cleaned_data, dims=2)) # get the means of each row
	gene_sds = vec(std(cleaned_data; dims=2)) # get the standard deviation of each row
	gene_cvs = gene_sds ./ gene_means # get the coefficient of variation (CV) -- CV is a good measure of relative variability

	criteria1 = findall(in(symbols_of_interest), data_symbol_list) # find genes list of known genes in full list of genes from data set
	criteria2 = findall(gene_means .> minMean) # find all genes with a mean expression level greater than the minimum (user defined) mean
	criteria3 = findall(gene_cvs .> minCV) # find all genes with a CV greater than the minimum (user defined) CV...
	criteria4 = findall(gene_cvs .< maxCV) # ...but smaller than the maximum (user defined) CV.

	allCriteria = intersect(criteria1, criteria2, criteria3, criteria4) # find all the genes for which all conditions are true
	seed_data = OG_data[allCriteria, :] # store all samples of genes that meat all the criteria in seed_data
	seed_symbols = data_symbol_list[allCriteria, :] # store all the names of the genes being kept in seed_symbols

	seed_symbols, seed_data # return the stored gene names and their respective samples
end
#--end function--#
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
# Max for bluntPercent
function removeOutliers!(data::Array{Float32, 2}, bluntPercent)
	ngenes, nsamples = size(data) # number of rows = ngenes, number of columns = nsamples
	nfloor = Int(1 + floor((1 - bluntPercent) * nsamples)) # index of lowest value to be kept
	nceiling = Int(ceil(bluntPercent*nsamples)) # index of highest value to be kept
	for row in 1:ngenes # go through each row
		sorted = sort(vec(data[row, :])) # sort each row from lowest to highest
		vfloor = sorted[nfloor] # find the value of the desired new minimum
		vceil = sorted[nceiling] # find the value of the desired new maximum
		for sample in 1:nsamples # go through each sample of each row
			data[row, sample] = max(vfloor, data[row, sample]) # if the value of a given field is lower than the new minimum it will be replaced with the new minimum
			data[row,sample] = min(vceil, data[row, sample]) # if the value of a given field is higher than the new maximum it will be replaced with the new maximum
		end #--end for--#
	end #--end for--#

	data
end
#--end function--#
########################
# end removeOutliers! #
######################



########################
# begin getEigengenes #
#####################################################################################################################################################################
# Convert data to SVD space, keeping eigengenes that contribute a minimum (user defined) amount of variance, up to minimum (user defined) total variance captured  #
###################################################################################################################################################################
# Method 1: Array{Float32,2}                                                                                                                                     #
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Max for total_fraction_var = 1, reasonable value for total_fraction_var = 0.97; Max for indiv_frac_var = 1, reasonable value for indiv_frac_var = 0.025-0.05 #
#-------------------------------------------------------------------------------------------------------------------------------------------------------------#
function getEigengenes(numeric_data::Array{Float32, 2}, total_fraction_var::Number, indiv_frac_var::Number, maxeig::Number=30)
    svd_obj = svd(numeric_data) # convert data to SVD object containing singular values and eigengene expression data in SVD space
    expvar = cumsum(svd_obj.S.^2, dims = 1) / sum(svd_obj.S.^2) # .S are the singular values, sorted in descending order. Find the Fraction variance that an eigengene and each eigengene before it makes up from the total variance.

    ReductionDim1 = 1 + length(expvar[expvar .<= total_fraction_var]) # how many eigengenes need to be included to have captured the minimum (user specified) variance from the eigengenes (from their singular values)
    vardif = diff(expvar, dims = 1) # find the difference between each added eigengene
    ReductionDim2 = 1 + length(vardif[vardif .>= indiv_frac_var]) # which eigengenes contribute the minimum (user specified) variance
    ReductionDim = min(ReductionDim1, ReductionDim2, maxeig) # The last criteria is the maximum number of eigengenes (user specified) that will be kept. Whichever is the smallest is the number of eigengenes kept

    Transform = svd_obj.V[:, 1:ReductionDim]' # .V is the expression of the eigengenes (in SVD space)

    ReductionDim, Array{Float32,2}(10*Transform) # Return the number of eigengenes kept, and the expression of the eigengenes.
end
#--end function--#
######################
# end getEigengenes #
####################



#--------------------#
# Data Manipulation #
#------------------#

end
#--end module--#
