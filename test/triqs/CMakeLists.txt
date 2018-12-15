copyfiles()

set(triqs_tests maxent_result analyzers set_G_tau set_G_tau_complex_matrix
    tau_maxent_off srvo3_mesh_and_ALPS complex_elementwise_maxent G_w_from_A_w
    G_tau_from_A_w sigma_continuation)

set(all_tests ${triqs_tests})

addtests()
