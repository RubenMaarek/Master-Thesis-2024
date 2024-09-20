# lib/config.py
from lib.alg import (
    vqe_cost_function, 
    vqe_cost_function_for_dmin, 
    vqe_cost_function_for_dmax,  
    vqe_sampling_cost_function,
    vqe_sampling_cost_function_for_dmin,
    vqe_sampling_cost_function_for_dmax,
    proba_loss,
    constrained_overlap_with_Pi, 
    constrained_distance_with_Pi, 
)
from typing import Union

print("config.py is being loaded")

cost_funcs = [vqe_cost_function, vqe_sampling_cost_function]

lp_boundary_funcs = [
    vqe_cost_function_for_dmin,
    vqe_sampling_cost_function_for_dmin,
    vqe_cost_function_for_dmax,
    vqe_sampling_cost_function_for_dmax,
]

performance_funcs = [constrained_distance_with_Pi]
print("performance_funcs defined in config.py:", performance_funcs)

