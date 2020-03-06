using ValueShapes
using ArraysOfArrays
using StatsBase 
using LinearAlgebra
using Statistics
using BAT
using Distributions 
using IntervalSets

using HCubature
using JLD
using CPUTime

# ***************************************
# Normal Distribtion: 
# ***************************************

PATH = "/home/iwsatlas1/vhafych/MPP-Project/AHMI_publication/NormalDistributionData/normal_dist-ffcor-3.jld" # path where data will be saved


HMI_Manual_Settings = BAT.HMISettings(BAT.cholesky_partial_whitening!, 
        1000, 
        1.5, 
        0.1, 
        true, 
        16, 
        true, 
        Dict("cov. weighted result" => BAT.hm_combineresults_covweighted!)
    )

nchains = 5
nsamples = 2*10^5

function run_integrations(dim_range::StepRange{Int64,Int64}, n_repeat::Int64; 
    nchains = nchains,
    nsamples = nsamples,
    HMI_Manual_Settings = HMI_Manual_Settings)
	
    # information that we want to track 
	integrals_ahmi_array = Vector{Float64}()
	integrals_true_array = Vector{Float64}()
	dim_array = Vector{Int64}()
	uns_ahmi_array = Vector{Float64}()
	mcmc_time_array = Vector{Float64}()
	ahmi_time_array = Vector{Float64}()
	n_samples_array = Vector{Tuple{Int64,Int64}}()
	tot_volumes_accepted1_array = Vector{Int64}()
    tot_volumes_rejected1_array = Vector{Int64}()
    tot_volumes_accepted2_array = Vector{Int64}()
    tot_volumes_rejected2_array = Vector{Int64}()
    
    int_estimates_1_array = Vector{Any}()
    int_estimates_2_array = Vector{Any}()
    
	for dim_run in dim_range
		
		@show dim_run

		prior = NamedTupleDist(a = [[Normal() for i in 1:dim_run]...],)
        
        ####### Problem-specific 
		integral_true_run = 0.0 
        ####### Problem-specific 

		for n_run in 1:n_repeat

			@show dim_run, n_run

			mcmc_ex_time = @CPUelapsed begin 
                samples = bat_sample(prior, nchains*nsamples).result;  # iid sampling from BAT
            end

			hmi_data = BAT.HMIData(unshaped.(samples))
			ahmi_ex_time = @CPUelapsed BAT.hm_integrate!(hmi_data, settings = HMI_Manual_Settings)

			ahmi_integral_run =[hmi_data.integralestimates["cov. weighted result"].final.estimate, hmi_data.integralestimates["cov. weighted result"].final.uncertainty]
			log_smpl_int = log.(ahmi_integral_run)
            
            push!(int_estimates_1_array, [hmi_data.integrals1.integrals])
            push!(int_estimates_2_array, [hmi_data.integrals2.integrals])
            
            push!(dim_array, dim_run)
			push!(n_samples_array, size(flatview(unshaped.(samples.v))))
            
			push!(tot_volumes_accepted1_array, length(hmi_data.volumelist1))
			push!(tot_volumes_accepted2_array, length(hmi_data.volumelist2))
			push!(tot_volumes_rejected1_array, length(hmi_data.rejectedrects1))
			push!(tot_volumes_rejected2_array, length(hmi_data.rejectedrects2))
			
			push!(mcmc_time_array, mcmc_ex_time)
			push!(ahmi_time_array, ahmi_ex_time)

			push!(integrals_ahmi_array, log_smpl_int[1])
			push!(integrals_true_array, integral_true_run)
			push!(uns_ahmi_array, log_smpl_int[2])

		end
		
		# Save all data after each dimension. This protects from losing data if AHMI/MCMC fails. 
		
		save_data(deepcopy(n_samples_array), 
            deepcopy(integrals_true_array), 
            deepcopy(integrals_ahmi_array), 
            deepcopy(uns_ahmi_array), 
            deepcopy(mcmc_time_array), 
            deepcopy(ahmi_time_array), 
            deepcopy(dim_array), 
            deepcopy(dim_run), 
            deepcopy(tot_volumes_accepted1_array), 
            deepcopy(tot_volumes_accepted2_array), 
            deepcopy(tot_volumes_rejected1_array), 
            deepcopy(tot_volumes_rejected2_array),
            deepcopy(int_estimates_1_array), 
            deepcopy(int_estimates_2_array))
		
    end
end

function save_data(n_samples_array::Vector{Tuple{Int64,Int64}}, 
        integrals_true_array::Vector{Float64}, 
        integrals_ahmi_array::Vector{Float64}, 
        uns_ahmi_array::Vector{Float64}, 
        mcmc_time_array::Vector{Float64}, 
        ahmi_time_array::Vector{Float64}, 
        dim_array::Vector{Int64}, 
        dim_run::Int64, 
        tot_volumes_accepted1_array::Vector{Int64}, 
        tot_volumes_accepted2_array::Vector{Int64}, 
        tot_volumes_rejected1_array::Vector{Int64}, 
        tot_volumes_rejected2_array::Vector{Int64},
        int_estimates_1_array::Vector{Any},
        int_estimates_2_array::Vector{Any};
        PATH=PATH)
    
	x_dms = Int64(length(integrals_ahmi_array)/length(unique(dim_array)))
    y_dms = Int64(dim_array[end]-dim_array[1]+1)
    
    n_samples_array = reshape(n_samples_array, x_dms, y_dms)
	integrals_true_array = reshape(integrals_true_array, x_dms, y_dms)
	integrals_ahmi_array = reshape(integrals_ahmi_array, x_dms, y_dms)
	
	tot_volumes_accepted1_array = reshape(tot_volumes_accepted1_array, x_dms, y_dms)
	tot_volumes_accepted2_array = reshape(tot_volumes_accepted2_array, x_dms, y_dms)
	tot_volumes_rejected1_array = reshape(tot_volumes_rejected1_array, x_dms, y_dms)
	tot_volumes_rejected2_array = reshape(tot_volumes_rejected2_array, x_dms, y_dms)
    
    int_estimates_1_array = reshape(int_estimates_1_array, x_dms, y_dms)
	int_estimates_2_array = reshape(int_estimates_2_array, x_dms, y_dms)
		
	uns_ahmi_array = reshape(uns_ahmi_array, x_dms, y_dms)
	mcmc_time_array = reshape(mcmc_time_array, x_dms, y_dms)
	ahmi_time_array = reshape(ahmi_time_array, x_dms, y_dms)
    dim_array = reshape(dim_array, x_dms, y_dms)
		
	integrals_ahmi_array = convert(Array{Float64,2}, integrals_ahmi_array)
	uns_ahmi_array = convert(Array{Float64,2}, uns_ahmi_array);
		
	isfile(PATH) && rm(PATH)
	@show "saving"
	save(PATH, 
		"sample_size", n_samples_array, 
		"integrals_ahmi_array", integrals_ahmi_array, 
		"integrals_true_array", integrals_true_array,  
		"uns_ahmi_array", uns_ahmi_array,  
		"dim_array", dim_array, 
		"mcmc_time_array", mcmc_time_array,
		"ahmi_time_array", ahmi_time_array,
		"tot_volumes_accepted1_array", tot_volumes_accepted1_array,
		"tot_volumes_accepted2_array", tot_volumes_accepted2_array, 
		"tot_volumes_rejected1_array", tot_volumes_rejected1_array, 
		"tot_volumes_rejected2_array", tot_volumes_rejected2_array,
        "int_estimates_1_array", int_estimates_1_array, 
		"int_estimates_2_array", int_estimates_2_array,
	)
end

dim_range = range(2, step=1, stop=31)

@CPUtime run_integrations(dim_range, 10)