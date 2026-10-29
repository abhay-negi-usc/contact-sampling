# contact-sampling

Contact-space sampling, contact-mode discovery, pose estimation and learned assembly for two part pairs: a
cylindrical peg/hole and a hook/eyelet. Scripts run from the repository root in the `mujoco_env` conda env
(`conda run -n mujoco_env python scripts/<script>.py --help`); each script's docstring describes its method and
usage. Generated data, videos and reports go to `/media/rp/Elements1/abhay_ws/contact-sampling/`, not the repo.

| Area | Scripts |
|---|---|
| Models and reference trajectories | `hook_eyelet_model`, `peg_hole_model`, `convex_decompose_meshes`, `refine_assembly_trajectory`, `optimize_max_separation_trajectory`, `assembly_trajectory_visual_verification`, `peg_hole_assembly_trajectory` |
| Peg/hole contact sampling, analytic modes | `peg_hole_analytic`, `peg_hole_contact_sampler`, `compare_samples_to_analytic`, `plot_contact_samples_cylindrical` |
| Hook separation and goal fields | `build_separation_field`, `goal_field`, `validate_goal_field`, `route_pockets`, `compact_goal_field` |
| Peg/hole mode discovery | `mode_discovery_bench`, `contact_graph_spectral`, `stratum_spectral`, `knn_pose_spectral`, `codim_aware_scores`, `strata_decomposition`, `cad_features`, `cad_contacts`, `cad_stratum_graph` |
| Peg/hole assembly-trajectory groups | `assembly_groups`, `assembly_tolerance_sweep`, `plot_assembly_groups`, `plot_assembly_mode_graphs`, `pcca` |
| Hook modes and routes | `hook_topology`, `hook_mode_graph`, `mode_graph_experiments`, `hook_assembly_routes`, `hook_contact_strata`, `hook_sided_modes`, `hook_warp_contact`, `warp_contact_sampling` |
| Hook RL under pose uncertainty, KUKA | `hook_rl_env`, `train_hook_rl`, `eval_hook_rl`, `contact_estimators`, `estimator_bench`, `kuka_hook_sim`, `kuka_hook_deploy` |
| Pose estimation from contacts | `contact_information`, `info_metric_bench`, `contact_gpu`, `pose_filters`, `pose_filter_bench`, `uq_analysis`, `pose_identifiability`, `probe_design` |

`research_notes/` and `reports/` hold literature reviews on separating stratified contact-mode data.
