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
    x./sqrt(sum(x .* x)) # Convert x and y coordinates to points on a cirlce
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
    L1out = m.L1(x[1:m.o]) # Function for linear (dense, or fully connected) input layer
    circout = m.c(L1out) # Circular bottleneck
    L2out = m.L2(circout) # Function for linear (dense, or fully connected) output layer
end
#--end method--#



#------------------------------------------#
# Function generate CYCLOPS v1 model      #
#----------------------------------------#
# Generate a trainable CYCLOPS v1 model #
#--------------------------------------#
function genCYCLOPSv1(td)
    [oldcyclops(Dense(td[2],2), circ, Dense(2,td[2]), td[2])]
end
#--end method--#
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
#--end method--#



#----------------------------------#
# Generate CYCLOPS v2 model       #
#--------------------------------#
# Generate a trainable v2 model #
#------------------------------#
function genCYCLOPSv2(td, w::Number=1)
    nbatches = td[1] - td[2] # How many one-hot rows are there. This works now, since batch is the only categorical variable being used
    #-------------------------------------------------#
    # Initialize encoding and decoding sparse layers #
    #-----------------------------------------------#---------------------------------------------------------------------------------------------------------------------#
    # INFO: reshape the data into "pages," where each page is a batch, such that the standard deviation can be found for each gene (row) in each batch (page) separately #
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    rowstd = Array{Float32,2}(reshape(std(reshape(td[3][1:td[2],:], td[2], :, nbatches), dims = 2), td[2], nbatches)) # std of each gene for each batch
    S1 = w .* maximum(rowstd, dims = 2) ./ rowstd # the differing standard deviations of eigengenes between batches is corrected by the weights of the sparse layers
    S2 = 1 ./ S1 # this correction is un-done in the decoding layer
    #----------------------------------------------------------#
    # Initialize bias for encoding and decoding sparse layers #
    #--------------------------------------------------------#----------------------------------------------------------------------------------------------#
    # INFO: reshape the data into "pages," where each page is a batch, such that the mean can be found for each gene (row) in each batch (page) separately #
    #-----------------------------------------------------------------------------------------------------------------------------------------------------#
    bmean = Array{Float32,2}(reshape(mean(reshape(td[3][1:td[2],:], td[2], :, nbatches), dims = 2), td[2], nbatches)) # mean of each gene for each batch
    b1 = -S1 .* bmean # the differences in mean of eigengenes between batches is corrected by the biases of the sparse layer, normalizing the means to 0
    b2 = -b1 # in the decoding layer this normalization is un-done

    return [cyclops(param(S1), param(b1), Dense(td[2],2+l), circ, Dense(2+l,td[2]), param(S2), param(b2), td[1], td[2])]
end
#--end function--#
###################
# end CYCLOPS v2 #
#################



#---------------------------#
# end Model Initialization #
#-------------------------#

#-----------------------------------------------------------------------------#

#-----------------#
# Model Training #
#---------------#



###################
# begin mytrain! #
#############################################################################################
# Custom train! function returns average loss of entire epoch and prints it to the console #
###########################################################################################
# Method 1                                                                               #
#---------------------------------------------------------------------------------------#
call(f, xs...) = f(xs...) # Define call function
runall(f) = f # Define runall function
runall(fs::AbstractVector) = () -> foreach(call, fs)
struct StopException <: Exception end # Define type StopException

function customTrain!(loss, ps, data, opt; cb = () -> ())
    lossrec =[] # Initialize loss record
    ps = Params(ps) # Make ps trainable parameters
    cb = runall(cb) #
    @progress for d in data
        try
            gs = gradient(ps) do
                loss(d...)
            end
            update!(opt, ps, gs)
            append!(lossrec, loss(d...))
            cb()
        catch ex
            if ex isa StopException
                break
            else
                rethrow(ex)
            end #--end if--#
        end #--end try--#
    end #--end for--#
    avg = mean(lossrec)
    println(string("Average loss this epoch: ", avg))

    avg
    end
#--end method--#
#################
# end mytrain! #
###############



#######################
# begin customEpochs #
############################################################
# Custom @epochs that stores the output from customTrain! #
##########################################################
# Method 1: Evaluates ex n-number of times              #
#------------------------------------------------------#
macro customEpochs(n, ex)
    return :(lossrecord = [];
    @progress for i = 1:$(esc(n))
        @info "Epoch $i"
        avgloss = $(esc(ex))
        append!(lossrecord, avgloss)
    end; #-- end for--#
    lossrecord = map(x -> data(x), lossrecord); # Extract numerical data from tracked data

    lossrecord) #--close return--#
end
#--end macro--#
#################
# end customEpochs #
###############



####################
# begin customEpochs2 #
########################################################################################################################
# Custom @epochs that stores the output from customTrain! and additionally ends session when "no improvement" is seen #
######################################################################################################################
# Method 1: Evaluates ex a maximum of n-number of times and returns lossrecord                                      #
#------------------------------------------------------------------------------------------------------------------#
# This will be the future @customEpochs if it is stable                                                           #
#----------------------------------------------------------------------------------------------------------------#
# INFO: If the average loss of an epoch increases int-number of times in a row, training is interrupted         #
#--------------------------------------------------------------------------------------------------------------#
macro customEpochs2(n, ex, int)
    return :(lossrecord = []; # Initialize loss record
    @progress for i = 1:$(esc(n))
        @info "Epoch $i"
        avgloss = $(esc(ex))
        if length(lossrecord) > $(esc(int)) && sum(diff(lossrecord[length(lossrecord)-$(esc(int))+1:end]).>0) == $(esc(int)) # ...
            break # ...if the lossrecord is longer than a given (user defined) length and the last n (user defined) epochs have increased in average loss, break
        end #--end if--#
        append!(lossrecord, avgloss)
    end; #-- end for--#
    lossrecord = map(x -> data(x), lossrecord); # Extract numerical data from tracked data

    lossrecord) #--close return--#
end
#--end macro--#
######################
# end customEpochs2 #
####################



#########################
# begin CYCLOPSdecoder #
###################################################
# Convert x and y points on a circle to an angle #
#################################################
# Method 1: tracked data to numerical data     #
#---------------------------------------------#
function CYCLOPSdecoder(data_matrix, model, n_circs::Integer)
    points = size(data_matrix, 2) # Determine the total number of samples
    phases = zeros(n_circs, points) # Initialize list of phases
    base = 0
    for circ in 1:n_circs # how many circular bottleneck layers are there (usually 1)
        for n in 1:points # for each sample
            pos = model(data_matrix[:, n]) # pos is a 2 element array
            phases[circ, n] = data(atan(pos[2 + base], pos[1 + base])) # the inverse tangent of the model output is an angle, stored to phases
        end #--end for--#
        base += 2 # move to next circular layer (will in a future version be replaced by map())
    end #--end for--#

    phases # return list of phases
end
#--end method--#
#######################
# end CYCLOPSdecoder #
#####################



#######################
# begin trainMetrics #
#########################################################
# Trains a model using customTrain! and @customEpochs  #
#######################################################
# Method 1-5:                                        #
#---------------------------------------------------#
# INFO: Meant for iterative use in multimodeltrain #
#-------------------------------------------------#
function trainMetrics(td, ms, tms, ferrors, mlosses, epochs::Int=100, jj::Int=1, ii::Int=1, nc=1; version::Int=2)

    m = ms[ii]
    loss(x)= Flux.mse(m(x), x[1:td[2]]) # define loss function used for training

    ################################
    # Rather barbaric case switch #
    ##############################
    if version == 2.1
        lossrecord = CYCLOPS_TrainingModule.@customEpochs epochs CYCLOPS_TrainingModule.customTrain!(loss, Flux.params(m.S1, m.b1, m.L1, m.L2, m.S2, m.b2), zip(td[4]), ADAM())
        sparse(x) = (x[1:m.o].*(m.S1*x[m.o+1:end]) + m.b1*x[m.o+1:end])
        linlayer(x) = m.L1(x)
        trainedmodel = Chain(sparse, linlayer, circ)
    elseif version == 2
        lossrecord = CYCLOPS_TrainingModule.@customEpochs epochs CYCLOPS_TrainingModule.customTrain!(loss, Flux.params(m.S1, m.b1, m.L1, m.L2, m.S2, m.b2), zip(td[4]), momentum())
        sparse(x) = (x[1:m.o].*(m.S1*x[m.o+1:end]) + m.b1*x[m.o+1:end])
        linlayer(x) = m.L1(x)
        trainedmodel = Chain(sparse, linlayer, circ)
    elseif version == 1
        lossrecord = CYCLOPS_TrainingModule.@customEpochs epochs CYCLOPS_TrainingModule.customTrain!(loss, Flux.params(m.L1, m.L2), zip(td[4]), Momentum())
        oldlinlayer(x) = m.L1(x[1:m.o])
        trainedmodel = Chain(oldlinlayer, circ)
    end

    estimated_phaselist = CYCLOPSdecoder(td[3], trainedmodel, nc)
    estimated_phaselist = mod.(estimated_phaselist .+ 2*pi, 2*pi)
    estimated_phaselist = estimated_phaselist[td[6]]

    shiftephaselist = CYCLOPS_PrePostProcessModule.best_shift_cos(estimated_phaselist, td[5], "hours")
    errors = CYCLOPS_CircularStatsModule.circularerrorlist(2*pi * td[5] / 24, shiftephaselist)
    hrerrors = reshape(mean(reshape((12/pi) * abs.(errors),:,1,td[1]-td[2]), dims=1),:)

    tms[ii,jj] = m
    ferrors[ii,jj] = hrerrors
    mlosses[ii,jj] = Float32(lossrecord[end])

    return tms, ferrors, mlosses
end
#--end method--#
#####################
# end trainMetrics #
###################



##########################
# begin multimodeltrain #
########################
#  #
#########################
# Method 1 - 5
#----
function multimodeltrain(td, w=1; reps::Int=5, epochs::Int=100, nc::Int=1, version::Int=2)

    if version == 2
        tms = fill(genCYCLOPSv2(td, size(w,1))[1], size(w,1),reps) # Array{cyclops,2} that will contain all trained models
    else
        tms = fill(genCYCLOPSv1(td)[1], 1, reps)
    end #--end if--#

    ferrors = fill(Array{Float32}(repeat([Inf], td[1]-td[2])), size(w,1), reps) # initialize ferrors (errors associated with lowest machine learning loss)
    mlosses = zeros(size(w,1), reps) # initialize mlosses (machine learning losses)

    for jj = 1:reps # create new models with scaling weights centered around w five times

        #####################
        # ugly case switch #
        ###################
        if version == 2
            ms = genCYCLOPSv2(td, w) # using fillcyclops to generate an array of models
        else
            ms = genCYCLOPSv1(td)
        end #--end if--#

        for ii = 1:size(ms,1) # train one model on each core. The number of models that can be trained at once depend on the number of weights

            println("##########")
            println(string("# ", (jj-1)*size(ms,1)+ii, "/", reps*size(ms,1), " #"))
            println("########")
            tms, ferrors, mlosses = trainMetrics(td, ms, tms, ferrors, mlosses, epochs, jj, ii, version=version)
        end #--end for--#
    end #--end for--#

    tms, ferrors, mlosses # trained models, mean error from time of day, average machine learning loss
end
#--end method--#
########################
# end multimodeltrain #
######################


#---------------------#
# end Model Training #
#-------------------#

end
#--end module--#
