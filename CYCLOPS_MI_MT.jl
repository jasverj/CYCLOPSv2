using Flux, Juno, DataFrames, Statistics, LinearAlgebra, StatsBase, MultivariateStats, Distributions, CSV

#-----------------------#
# Model Initialization #
#---------------------#

###############
# begin circ #
##############################
# Circular bottleneck layer #
############################
function circ(x)
    length(x) == 2 || throw(ArgumentError(string("Invalid length of input that should be 2 but is ", length(x))))
    x./sqrt(sum(x .* x))
end
#--end function--#
#############
# end circ #
###########


#####################
# begin CYCLOPS v1 #
######################
# CYCLOPS v1 model  #
####################
# Type CYCLOPS v1 #
#----------------#
struct CYCLOPSv1
    L1  # Linear layer from # eigengenes to 2
    c   # Circ layer
    L2  # Linear layer from 2 to # eigengenes
    o   # output dimension used for oldcyclops
end
#--end type--#


#----------------------#
# Function CYCLOPS v1 #
#--------------------#
# Overload call     #
#------------------#
function (m::CYCLOPSv1)(x)
    L1out = m.L1(x[1:m.o])
    circout = m.c(L1out)
    L2out = m.L2(circout)
end
#--end function--#


#------------------------------------------#
# Function generate CYCLOPS v1 model      #
#----------------------------------------#
# Generate a trainable CYCLOPS v1 model #
#--------------------------------------#
function genCYCLOPSv1(td)
    [oldcyclops(Dense(td[2],2), circ, Dense(2,td[2]), td[2])]
end
#--end function--#
###################
# end CYCLOPS v1 #
#################



#####################
# begin CYCLOPS v2 #
######################
# CYCLOPS v2 model  #
####################
# Type CYCLOPS v2 #
#----------------#
struct cyclops
    S1  # Scaling factor for OH (encoding). Could be initialized as all ones.
    b1  # Bias factor for OH (encoding). Should be initialized as random numbers around 0.
    L1  # First linear layer (Dense). Reduced to at least 2 layers for the circ layer but can be reduced to only 3 to add one linear/non-linear layer.
    C   # Circular layer (circ(x))
    L2  # Second linear layer (Dense). Takes output from circ and any additional linear layers and expands to number of eigengenes
    S2  # Scaling factor for OH (decoding). Could be initialized as all ones
    b2  # Bias factor for OH (decoding). Should be initialized as random number around 0.
    i   # input dimension (in)
    o   # output dimensions (out)
end
#--end type--#


#----------------------#
# Function CYCLOPS v2 #
#--------------------#
# Overload call     #
#------------------#
function (m::cyclops)(x)
    SparseOut = (x[1:m.o].*(m.S1*x[m.o + 1:end]) + m.b1*x[m.o + 1:end]) #
    DenseOut = m.L1(SparseOut)
    CircOut = m.C(DenseOut)
    Dense2Out = m.L2(CircOut)
    SparseOut = (Dense2Out.*(m.S2*x[m.o + 1:end]) + m.b2*x[m.o + 1:end])
end
#--end function--#


#----------------------------------#
# Generate CYCLOPS v2 model       #
#--------------------------------#
# Generate a trainable v2 model #
#------------------------------#
function genCYCLOPSv2(td, w::Number=1)
    nbatches = td[1] - td[2] # How many one-hot rows are there. This works now, since batch is the only categorical variable being used
    #-------------------------------------------------#
    # Initialize encoding and decoding sparse layers #
    #-----------------------------------------------#--------------------------------------------------------------------------------------------------------#
    # INFO: reshape the data into "pages," where each page is a batch, such that the standard deviation can be found for each gene in each batch separately #
    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    rowstd = Array{Float32,2}(reshape(std(reshape(td[3][1:td[2],:], td[2], :, nbatches), dims = 2), td[2], nbatches)) # std of each gene for each batch
    S1 = w .* maximum(rowstd, dims = 2) ./ rowstd # the differing standard deviations of eigengenes between batches is corrected by the weights of the sparse layers
    S2 = 1 ./ S1 # this correction is un-done in the decoding layer
    #----------------------------------------------------------#
    # Initialize bias for encoding and decoding sparse layers #
    #--------------------------------------------------------#--------------------------------------------------------------------------------#
    # INFO: reshape the data into "pages," where each page isa batch, such that the mean can be found for each gene in each batch separately #
    #---------------------------------------------------------------------------------------------------------------------------------------#
    bmean = Array{Float32,2}(reshape(mean(reshape(td[3][1:td[2],:], td[2], :, nbatches), dims = 2), td[2], nbatches)) # mean of each gene for each batch
    b1 = -S1 .* bmean # the differences in mean of eigengenes between batches is corrected by the biases of the sparse layer, normalizing the means to 0
    b2 = -b1 # in the decoding layer this normalization is un-done

    return [cyclops(param(S1), param(b1), Dense(td[2],2+l), circ, Dense(2+l,td[2]), param(S2), param(b2), td[1], td[2])]
end
#--end function--#


end
#--end module--#
