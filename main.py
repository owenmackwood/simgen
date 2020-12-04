import numpy as np
from pathlib import Path
from inner import inner_c, InterneuronPlasticity, PlasticitySubtype
from utils import dtype, allocate_aligned, reallocate_aligned, compute_connections_sparse

sp_type = InterneuronPlasticity.EXC2INH
sp_subtype = PlasticitySubtype.FULL_GRADIENT
i2e_plastic = True

n_e = 512
n_i = n_e // 4
n_steps = 10**6

e2e = compute_connections_sparse(np.infty, 0.1, n_e, n_e, True, False)
i2i = compute_connections_sparse(np.infty, 0.1, n_i, n_i, True, False)
e2i = compute_connections_sparse(np.infty, 0.1, n_e, n_i, False, False)
i2e = compute_connections_sparse(np.infty, 0.1, n_i, n_e, False, False)

inner_c(
    n_steps=n_steps, dt=5e-3,
    tau_exc=50e-3, tau_inh=25e-3,
    n_e=n_e, n_i=n_i, alpha_r=1., i2e_binary=False,
    sp_type=sp_type, sp_subtype=sp_subtype, calibrate_syn="", f_max=100.,
    rho0=0.1, rho_calibrate=2., alpha_p=1.,
    in_exc_mean=reallocate_aligned(np.random.uniform(4., 6., n_e).astype(dtype)),
    in_exc_state=allocate_aligned(n_e, dtype=dtype),
    in_ou_exc_sigma=2., in_ou_exc_tau=1.,
    e2e_data=e2e.csr_indices.data, e2e_indptr=e2e.csr_indices.indptr, e2e_indices=e2e.csr_indices.indices,
    i2i_data=i2i.csr_indices.data, i2i_indptr=i2i.csr_indices.indptr, i2i_indices=i2i.csr_indices.indices,
    e2i_data=e2i.csr_indices.data, e2i_indptr=e2i.csr_indices.indptr, e2i_indices=e2i.csr_indices.indices,
    i2e_data=i2e.csr_indices.data, i2e_indptr=i2e.csr_indices.indptr, i2e_indices=i2e.csr_indices.indices,
    e2i_data_i=e2i.row_ind, e2i_data_j=e2i.col_ind,
    i2e_data_i=i2e.row_ind, i2e_data_j=i2e.col_ind,
    steps_per_frame=10 ** 3,
    update_weights_n=10,
    plasticity_on=0, plasticity_off=n_steps,
    seed=0, build_path=Path("./output").absolute(), i2e_plastic=i2e_plastic,
    do_print_arrays=True, rec_all_plastic=False, do_mmap=True
)
