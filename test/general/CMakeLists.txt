copyfiles()

set(general_tests omega_meshes minimizer_cubefun minimizer_circlefun
    minimizer_sin cached_view plus_minus_entropy maxent_cost_function_lazy
    maxent_cost_function_d functions_different_grid tau_kernel alpha_meshes
    default_models logtaker matrix_maxent_result maxent_loop tau_maxent
    bryan_cost_function huge_alpha cov elementwise_maxent elementwise_set_G
    pickle_maxent_result complex_A srvo3_mesh_and_ALPS)

set(all_tests ${general_tests})

addtests()
