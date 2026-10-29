# Trajectory- and Dynamics-Based Separation of Contact Modes (MSMs, switching models, changepoints, hybrid/PWA identification, piecewise-continuous regression)

Scope note for the report writer: every item under "Cited Findings" has an inline source. Application notes for the user's data are in "Inferences" (they are my reasoning, not literature claims). Repo context used in the inferences (from the module docstrings of `scripts/peg_hole_contact_sampler.py`, `scripts/contact_graph_spectral.py`, `scripts/stratum_spectral.py`): each peg-in-hole sample row has `pose` (7), `wrench` (6, peg frame, about the peg origin, with sliding friction at speed `--speed` and press `--press-n`), `twist` (6, walk velocity, hole frame), `traj`/`step`/`time`, a parent link (`parent_traj`, `parent_step`) so each worker's trajectories form one connected tree, `coords`, `n_active`, and per-contact `point`/`normal`/`features`. The walk moves inside a stratum by tangent steps in ker J_A followed by Newton projection; a step "makes" a contact when it closes another gap and "breaks" one when the target pulls away, but never breaks the last one. Coverage is RRT-driven (targets sampled in a box, extensions from the nearest node), so the process is neither reversible nor stationary.

## 1. Markov state models, metastable decomposition, and how to handle a non-reversible / non-equilibrium sampler

### Takeaway
MSM machinery (microstates -> transition matrix -> PCCA+/GPCCA or VAMP/VAMPnets) is the right tool when modes are *kinetically* separated (rarely crossed boundaries): it fits hook-in-eyelet (narrow passages) almost perfectly, and peg-in-hole at the "family" level. For a non-equilibrium, branching RRT walk, use estimators that do not assume detailed balance (GPCCA via Schur vectors, VAMP / Koopman SVD, generalized MSMs) or regenerate dedicated reversible walks. For peg-in-hole *mode*-level separation, plain transition kinetics on pose cells is weak because most mode boundaries are not bottlenecks; the fix is to build microstates or edge weights that encode the pose->wrench discontinuity so that mode boundaries become artificial bottlenecks.

### Cited Findings

#### Core metastability / spectral machinery
- PCCA+ (Deuflhard & Weber 2005, "Robust Perron cluster analysis in conformation dynamics", Linear Algebra Appl. 398:161–184) forms fuzzy membership vectors as a linear transformation of the dominant eigenvectors of the transition matrix into the unit simplex (almost-characteristic functions of metastable sets) — [Deuflhard & Weber 2005](https://doi.org/10.1016/j.laa.2004.10.026)
- Röblitz & Weber (2013) present PCCA+ as a general fuzzy spectral clustering method for MSMs *and* for generic data classification — [Röblitz & Weber 2013, Adv. Data Anal. Classif.](https://doi.org/10.1007/s11634-013-0134-6)
- Normalized-cut spectral clustering has an exact random-walk interpretation: minimizing the normalized cut is equivalent to finding sets with low escape probability for the random walk P = D^-1 W; the generalized eigenvectors of (W, D) are eigenvectors of P — [Meilă & Shi 2001, "A random walks view of spectral segmentation", AISTATS](https://proceedings.mlr.press/r3/meila01a.html)
- Diffusion maps define multiscale embeddings from the eigenfunctions of a Markov kernel on the data; Nadler et al. link diffusion maps, spectral clustering, and reaction coordinates / metastable states of stochastic dynamical systems — [Coifman & Lafon 2006, ACHA](https://doi.org/10.1016/j.acha.2006.04.006); [Nadler, Lafon, Coifman, Kevrekidis 2006, arXiv math/0503445](https://arxiv.org/abs/math/0503445)
- TICA finds linear combinations of features that maximize time-lagged autocorrelation (slowest linear coordinates) and was introduced for MSM construction — [Pérez-Hernández et al. 2013, arXiv 1302.6614](https://arxiv.org/abs/1302.6614)
- Core-set MSMs / milestoning: trajectories are assigned to "cores" (well inside metastable sets) and transition regions are assigned by the last core visited / committor, giving MSMs whose quality depends less on boundary discretization — [Schütte, Noé, Lu, Sarich, Vanden-Eijnden 2011, "Markov state models based on milestoning", J. Chem. Phys. 134, 204105](https://doi.org/10.1063/1.3590108)
- Transition path theory (committors q(x) = probability of hitting B before A; reactive flux) for Markov jump processes — [Metzner, Schütte, Vanden-Eijnden 2009, Multiscale Model. Simul.](https://doi.org/10.1137/070699500)

#### Learned / variational (Koopman) approaches
- VAMP (variational approach for Markov processes) approximates the *singular* functions of the Koopman/transfer operator and is valid for non-reversible and non-stationary dynamics (unlike TICA/VAC, which assume reversibility) — [Wu & Noé, "Variational approach for learning Markov processes from time series data", arXiv 1707.04659](https://arxiv.org/abs/1707.04659)
- VAMPnets learn the whole MSM pipeline end to end: a neural network maps coordinates to soft (softmax) Markov-state memberships, trained by maximizing a VAMP score — [Mardt, Pasquali, Wu, Noé 2018, Nat. Commun. 9:5](https://www.nature.com/articles/s41467-017-02388-1)
- State-free reversible VAMPnets (SRVs) learn nonlinear slow modes (eigenfunctions) rather than discrete states, for reversible dynamics — [Chen, Sidky, Ferguson 2019, arXiv 1902.03336](https://arxiv.org/abs/1902.03336)
- Koltai, Wu, Noé, Schütte (2018) treat equilibrium and non-equilibrium cases (time-varying forcing, transient relaxation, no equilibrium distribution) in one framework of optimal low-rank transfer-operator approximation ("generalized MSMs") and discuss estimation from finite simulation data — [Koltai et al. 2018, Computation 6(1):22](https://publications.imp.fu-berlin.de/2176) (preprint arXiv 1801.04254)

#### Non-reversible chains: GPCCA
- Generalized PCCA (G-PCCA) replaces PCCA+'s eigenvalue problem by a *real Schur decomposition*, so it works for non-reversible chains (complex spectra) and does not require a positive stationary distribution; besides metastable sets it can identify dominant cycles and sinks — [pyGPCCA documentation](https://pygpcca.readthedocs.io/); [Fackeldey et al., "Spectral Clustering for Non-Reversible Markov Chains" (ZIB report)](https://opus4.kobv.de/opus4-zib/frontdoor/index/index/docId/7040)
- pyGPCCA sorts the m dominant eigenvalues (default: largest magnitude) to the top-left of the Schur form and returns that block — [pyGPCCA source docs](https://pygpcca.readthedocs.io/en/latest/_modules/pygpcca/_gpcca.html); package on [PyPI](https://pypi.org/project/pygpcca)
- MD applications: amyloid-β under an oscillating electric field (non-equilibrium, non-autonomous driving) — Reuter et al. 2018 JCTC, cited by [pyGPCCA docs](https://pygpcca.readthedocs.io/); follow-up "Generalized Markov modeling of nonreversible molecular kinetics" — [Reuter, Fackeldey, Weber 2019, J. Chem. Phys. 150, 174103](https://doi.org/10.1063/1.5064530)
- CellRank (single-cell biology) is a large-scale, non-MD user of GPCCA: "kernels" build a cell–cell transition matrix (VelocityKernel from RNA-velocity vectors, PseudotimeKernel, etc.) and the recommended GPCCA estimator returns soft macrostate memberships plus a coarse-grained transition matrix, and classifies initial/intermediate/terminal states and fate probabilities — [CellRank GPCCA estimator docs](https://cellrank.readthedocs.io/en/stable/api/_autosummary/estimators/cellrank.estimators.GPCCA.html); [CellRank initial/terminal states tutorial](https://cellrank.readthedocs.io/en/stable/notebooks/tutorials/estimators/600_initial_terminal.html); paper: [Lange et al. 2022, Nat. Methods](https://doi.org/10.1038/s41592-021-01346-6)

#### Trajectory-based coherent sets from sparse trajectories
- Banisch & Koltai extend diffusion maps to *trajectory space*: they construct a random walk on data points that combines the time ordering of trajectories and spatial closeness, giving "dynamical coordinates" that reveal coherent sets; the only prior needed is a locally valid distance; in the infinite-data limit it converges to Froyland's transfer-operator framework for coherent sets; designed for sparse/incomplete trajectory data — [Banisch & Koltai 2017, Chaos 27, 035804 (arXiv 1603.04709)](https://arxiv.org/abs/1603.04709)
- Network-based measures on Lagrangian trajectories remain effective when data are sparse — [Banisch, Koltai, Padberg, arXiv 1901.00799](https://www.arxiv.org/pdf/1901.00799)

#### Software
- deeptime: general Python library for dynamical models from time series — MSMs, Koopman models, VAMP, VAMPnets, deep MSMs; scikit-learn-style estimators; MSM objects provide free energies, relaxation times and transition paths; conda-forge recommended — [deeptime on PyPI](https://pypi.org/project/deeptime/0.4.2/); paper [Hoffmann et al., arXiv 2110.15013](https://arxiv.org/abs/2110.15013v1)
- pyGPCCA (PyPI/conda, documented) — [pyGPCCA docs](https://pygpcca.readthedocs.io/); CellRank — [docs](https://cellrank.readthedocs.io/en/latest/api/_autosummary/estimators/cellrank.estimators.GPCCA.html)

### Inferences

**Why the user's cell-graph spectral clustering stalls at mode level (kinetic vs. regime partitions).** By Meilă & Shi, spectral clustering of the trajectory-transition cell graph *is* metastable-set detection for the random walk on that graph. It finds sets the walk rarely leaves. On the peg-in-hole contact surface most mode boundaries are not rarely crossed: the walk slides from one single-contact sheet to the next without any bottleneck, and pose is continuous there. So the leading eigenvectors mostly carry the large-scale geometry of the manifold (its slow diffusion modes), not the modes. The stratum-graph experiment works (ARI ~0.85) because collapsing samples onto analytic modes leaves only the *between-mode* flux, and that flux has community structure at the "family" level. Conclusion: MSM tools need a state space where mode boundaries *are* bottlenecks.

**How to make mode boundaries kinetic bottlenecks without labels:**
1. *Augmented-space microstates.* Run k-means or regular-space clustering on [reduced pose (rho, radial tilt, |tangential tilt|, depth), lambda*ŵ], where ŵ is a friction-corrected unit wrench or the unit contact normal (see §5 on friction). Then count trajectory transitions between microstates (including parent->child first steps). A cell that straddles a jump boundary splits into two microstates with different ŵ, and transitions between them are rare relative to within-mode transitions. This is the unsupervised analogue of the stratum graph. It matches the user's finding that wrench features lift ARI to ~0.85, and it adds kinetics. Apply PCCA+ (for k use crispness / minChi, not just the eigengap) or GPCCA on the row-normalized count matrix.
2. *Edge-aware transition weights.* Keep the pose cells but weight each trajectory step i->i+1 by exp(-angle(ŵ_i, ŵ_{i+1})^2 / sigma^2), or by 0 if a changepoint detector (§3) fires between them. This is the transition-graph analogue of intensity-weighted normalized cuts. Spectral clustering / PCCA+ on this graph then sees the jump boundaries as cuts.
3. *Tangent-continuous boundaries* (an edge sliding round a rim, where the normal turns smoothly) show up in neither (1) nor (2). Add a curvature-like feature such as kappa = |Δŵ| / |Δq| along the trajectory (the rate at which the normal turns). It is piecewise smooth with jumps at C1 boundaries (§3, §5).

**Handling the non-equilibrium RRT walk.**
- The sampler's transitions are target-driven and branching. Transition counts reflect exploration order and density, not an equilibrium measure, and detailed balance fails. Practical options, from cheapest up: (a) symmetrize counts (C + C^T), which is what an undirected cell graph implicitly does. That is fine for finding cuts but gives no physical rates. (b) Use row-normalized directed counts with GPCCA (Schur vectors): no stationarity assumption, and it reveals sinks (for example deep-in-bore regions the RRT drifts into) and cycles. (c) Use VAMP / VAMPnets, which estimate singular functions of the forward/backward operators from non-stationary data, so a VAMPnet with k softmax outputs on (pose, ŵ) features gives soft modes directly. (d) For clean kinetics, generate separate short unbiased random walks: fixed step distribution, no target bias. In free space, a symmetric ball walk with collision rejection is reversible with respect to the uniform measure on C-free. On the contact manifold, a projection-based walk needs a Metropolis / reverse-projection correction to be reversible (cf. the manifold MCMC of Zappa, Holmes-Cerfon & Goodman, arXiv 1702.08446; I did not verify the reverse-check detail from the abstract).
- The parent links make each worker's data a tree of trajectories. For count estimation, treat each root-to-leaf path as a trajectory, or count each step pair once. Count the first step from the parent sample as a transition, as the repo already does.

**Hook-in-eyelet is the textbook MSM case.** Free-space regions separated by narrow passages are metastable sets for any local random walk. Recipe:
- Microstates by regular-space or k-means clustering in a pose metric. Use position plus a rotation metric: quaternion with sign fixed, or rotation-vector chart patches.
- Lag-τ count matrix from the walk, then implied timescales to choose τ and k.
- PCCA+/GPCCA memberships, then TPT committors between metastable sets. Committor ≈ 0.5 samples locate the narrow passages, the "transition states".

Core-set milestoning (Schütte et al.) handles fuzzy passage samples well: it assigns each sample to the last core visited along its trajectory. deeptime covers all of these steps. A VAMPnet gives a pose->mode classifier that generalizes to new poses.

**Core-set idea for peg-in-hole.** Define "cores" as samples far from any detected changepoint (or edge-indicator, §5). Assign boundary samples via the last core visited along the trajectory, or via a committor between cores computed on the trajectory graph. This removes the boundary-cell contamination that hurts cell-majority labelling.

**CellRank's VelocityKernel is a direct template for using the `twist` field.** It builds a non-reversible transition matrix on a kNN graph from the cosine between each sample's velocity and the displacement to each neighbour. It uses no time ordering at all, so it can densify the sparse trajectory graph with kNN edges that respect the observed direction of motion. GPCCA then extracts macrostates.

**Scalability.** Microstate counts of 10^3–10^5 are routine for MSM software. Millions of samples only enter through clustering and counting, which are linear. Sparse dominant eigensolves / Schur decompositions of a 10^5-state matrix are feasible. VAMPnets train by SGD on time-lagged pairs, so millions of pairs are fine.

### Gaps
- I did not verify current release dates / maintenance status for deeptime, pyGPCCA or CellRank. PyEMMA is widely described as superseded by deeptime, and MSMBuilder as no longer developed, but I did not confirm either from a primary source in this session.
- I did not verify whether pyGPCCA's sparse Schur path (SLEPc/PETSc) scales to more than 10^5 states.
- I found no paper that applies MSM/PCCA+ to contact-mode discovery in manipulation. The mapping above is my own.
- Theory for MSMs on *stratified* spaces (unions of manifolds of different dimension, as with "sticky" sphere systems) exists in the sticky-sphere literature (e.g., Zappa, Holmes-Cerfon & Goodman, [arXiv 1702.08446](https://arxiv.org/abs/1702.08446)). From the abstract alone I could not confirm that their sampler handles different-dimensional strata or includes a reversibility check.

## 2. Representation learning from trajectories (slow features, time-contrastive learning, Laplacian / successor representations, proto-value functions)

### Takeaway
These methods learn an embedding in which temporally close states are close and rarely connected states are far apart. They are parametric, scalable stand-ins for transition-graph spectral clustering, and they generalize to unseen poses. They separate *bottleneck-delimited* regions (hook-in-eyelet), not dynamical regimes, unless the "time" signal is designed to change at mode boundaries. Time-contrastive learning, trained on changepoint-delimited segments, is the variant most aligned with regime separation.

### Cited Findings
- Slow feature analysis (SFA) extracts features that vary as slowly as possible over time, subject to unit-variance and decorrelation constraints — [Wiskott & Sejnowski 2002, Neural Computation 14(4)](https://doi.org/10.1162/089976602317318938)
- Time-contrastive learning (TCL): divide a multivariate time series into segments and train a multinomial classifier to predict the segment index. The hidden representation then performs nonlinear ICA of nonstationary sources, which is identifiable under nonstationarity of the source distributions across segments — [Hyvärinen & Morioka 2016, NeurIPS, arXiv 1605.06336](https://arxiv.org/abs/1605.06336)
- Time-lagged autoencoders learn nonlinear slow collective variables by reconstructing x_{t+τ} from x_t — [Wehmeyer & Noé 2018, arXiv 1710.11239](https://arxiv.org/abs/1710.11239)
- The State Predictive Information Bottleneck (SPIB) learns a low-dimensional representation and a set of discrete metastable states jointly from MD trajectories, by predicting the future state through an information bottleneck — [Wang & Tiwary 2021, arXiv 2011.10127](https://arxiv.org/abs/2011.10127)
- Proto-value functions: eigenvectors of the graph Laplacian of the state-transition graph serve as basis functions that reflect the topology of the state space (bottlenecks, rooms) — [Mahadevan & Maggioni 2007, JMLR 8](https://jmlr.org/papers/v8/mahadevan07a.html)
- Eigenoptions: options derived from Laplacian eigenvectors (proto-value functions) discover temporally extended behaviours that traverse bottlenecks — [Machado, Bellemare, Bowling 2017, ICML, arXiv 1703.00956](https://arxiv.org/abs/1703.00956). A follow-up obtains eigenoptions from a *deep successor representation* learned from raw observations, connecting successor-representation (SR) eigenvectors to Laplacian eigenvectors — [Machado et al. 2018, ICLR, arXiv 1710.11089](https://arxiv.org/abs/1710.11089). The JMLR overview "Temporal Abstraction in RL with the Successor Representation" — [Machado, Barreto, Precup, Bowling, arXiv 2110.05740](https://arxiv.org/abs/2110.05740)
- Laplacian representations can be learned with neural networks from sampled transitions using a spectral graph-drawing objective, with no explicit graph — [Wu, Tucker, Nachum 2019, ICLR, arXiv 1810.04586](https://arxiv.org/abs/1810.04586); improved by a generalized graph-drawing objective that recovers individual eigenvectors — [Wang et al. 2021, ICML, arXiv 2107.05545](https://arxiv.org/abs/2107.05545)
- Earlier deep approximations depend on hard-to-tune hyperparameters, converge to arbitrary rotations of the eigenvectors, and do not recover eigenvalues. The Augmented Lagrangian Laplacian Objective (ALLO) recovers the true eigenvectors and eigenvalues without that hyperparameter dependence, with theoretical guarantees — [Gomez, Bowling, Machado 2024, "Proper Laplacian Representation Learning", ICLR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/7c88ed03238db64cef6f23bd0d6b5d11-Abstract-Conference.html) ([arXiv 2310.10833](https://arxiv.org/abs/2310.10833v2))

### Inferences
- **What "time" should mean here.** Using raw step order makes SFA, the Laplacian and the SR learn the slow geometric coordinates of the contact manifold, as with the cell-graph spectra. The useful trick is to change the notion of "neighbour in time":
  - *SFA / Laplacian with edge-aware pairs.* Train ALLO or graph-drawing on transition pairs (x_t, x_{t+1}), but drop or down-weight pairs that straddle a wrench jump or a detected changepoint. The learned eigenfunctions then become near-piecewise-constant on modes. This is the parametric version of the edge-aware transition graph in §1. ALLO is attractive because it returns actual eigenvalues, so an eigengap is still available for choosing k.
  - *TCL on segments.* After changepoint segmentation (§3), treat each segment as a TCL "class" and train an MLP from (reduced pose, ŵ, twist direction) to segment index. The penultimate-layer features are invariant within segments and discriminative across them. Clustering them (k-means, HDBSCAN) groups segments from different trajectories that share a mode. TCL's identifiability rests on distributional nonstationarity across segments, and a mode change in the pose->wrench law is exactly such nonstationarity.
  - *Time-lagged autoencoder / SPIB* with inputs x_t = (pose, ŵ, twist): SPIB's discrete output can stand in for VAMPnet-style soft modes, and it chooses the number of states itself.
- **Inputs:** pairs or sequences (x_t, x_{t+τ}) from `traj`/`step`, plus parent->child pairs. Embeddings come out per sample and generalize to new poses, which lets the user label held-out samples or new sampler runs.
- **Hook-in-eyelet:** eigenoptions/PVF results are literally about rooms and doorways. Laplacian eigenfunctions of the free-space walk will be near-constant per free-space region and change sharply across narrow passages. Sign patterns of the first k eigenfunctions (or k-means on them) give the modes.
- **Scalability:** all of these are SGD methods over transition pairs and scale to millions of samples on one GPU. Their main risk is the rotation/identifiability problem, which ALLO addresses.

### Gaps
- I found no application of TCL, SFA or Laplacian representation learning to contact-mode discovery. The suggestions above are untested.
- I did not survey code availability for ALLO or SPIB. Reference implementations are likely on the authors' GitHub, but this was not verified.

## 3. Trajectory segmentation and switching models (changepoints, BP-AR-HMM, sticky HDP-HMM, HDP-HSMM, (r)SLDS), and segment-then-cluster pipelines

### Takeaway
Detecting mode changes along each 1-D trajectory is far easier and better posed than detecting boundaries in scattered 4-D/6-D data. Each trajectory is a continuous slide, so a mode change is a changepoint in the local law: a jump in wrench/normal direction, a kink in its rate of change, or a change in the rank of the tangent/twist subspace. Penalized offline changepoint detection (PELT via `ruptures`) per trajectory, then clustering of segments, is the most scalable route. Bayesian switching models (BP-AR-HMM, HDP-HSMM, rSLDS) can share modes across trajectories and infer the mode count, but they do not scale directly to millions of samples. Contact-state HMMs and particle filters have a long history in assembly (peg-in-hole).

### Cited Findings

#### Changepoint detection
- Bayesian online changepoint detection (BOCPD) maintains a posterior over the "run length" since the last changepoint by exact message passing with conjugate predictive models — [Adams & MacKay 2007, arXiv 0710.3742](https://arxiv.org/abs/0710.3742)
- PELT finds the exact optimum of a penalized segmentation cost with expected computational cost linear in the data length (under conditions such as the number of changepoints growing with n) — [Killick, Fearnhead, Eckley 2012, JASA, arXiv 1101.1438](https://arxiv.org/abs/1101.1438)
- `ruptures` (Python) implements offline changepoint search methods (dynamic programming, PELT, binary segmentation, bottom-up, window) with interchangeable cost functions, including kernel and linear-model costs; the companion review is [Truong, Oudre, Vayatis 2020, "Selective review of offline change point detection methods", Signal Processing (arXiv 1801.00826)](https://arxiv.org/abs/1801.00826)
- Kernel multiple change-point detection with model selection detects changes in the full distribution, not only the mean — [Arlot, Celisse, Harchaoui, arXiv 1202.3878 (JMLR 2019)](https://arxiv.org/abs/1202.3878)
- ℓ1 trend filtering gives piecewise-linear fits whose kinks are chosen adaptively, i.e., it detects slope changes — [Kim, Koh, Boyd, Gorinevsky 2009, SIAM Review](https://doi.org/10.1137/070690274); the generalized-lasso view covers fused lasso on chains, grids and arbitrary graphs and polynomial trend filtering — [Tibshirani & Taylor, arXiv 1005.1971](https://arxiv.org/abs/1005.1971)
- CHAMP (Changepoint detection using Approximate Model Parameters) is an online Bayesian changepoint method that only requires fitting model parameters to data segments. It was applied to articulation models: objects whose articulation changes over time, *object–object contact producing quasi-static articulated motion*, and assembly tasks where each step changes the articulation relationship. A ROS `changepoint` package implements it with Gaussian and articulation-model plugins — [CMU RI tech report CMU-RI-TR-14-10](https://www.ri.cmu.edu/publications/champ-changepoint-detection-using-approximate-model-parameters); [Niekum, Osentoski, Atkeson, Barto 2015, "Online Bayesian Changepoint Detection for Articulated Motion Models" (PDF)](https://people.cs.umass.edu/~sniekum/pubs/CPD15.pdf); [ROS changepoint package](https://mirror.umd.edu/roswiki/changepoint.html)

#### Switching / nonparametric HMM models
- BP-AR-HMM: a beta-process prior lets many time series share a library of autoregressive "behaviours", with each series using a subset. The number of behaviours and the sharing pattern are inferred, using MCMC with split-merge and data-driven birth/death moves (no truncation). Demonstrated on unsupervised segmentation of CMU motion capture — [Fox, Hughes, Sudderth, Jordan 2014, Ann. Appl. Stat. 8(3):1281–1313, arXiv 1308.4747](https://arxiv.org/abs/1308.4747); earlier [NeurIPS 2009 version](https://papers.nips.cc/paper_files/paper/2009/file/4b04a686b0ad13dce35fa99fa4161c65-Paper.pdf)
- Review of Bayesian nonparametric Markov switching processes (sticky HDP-HMM, HDP-AR-HMM, HDP-SLDS, BP-AR-HMM) — [Fox, Sudderth, Jordan, Willsky, IEEE Signal Processing Magazine 2010 (PDF)](https://people.eecs.berkeley.edu/~jordan/papers/fox-etal-ieeespm10.pdf)
- HDP-HSMM adds explicit state-duration distributions to the HDP-HMM, which avoids rapid state switching and fits non-geometric durations — [Johnson & Willsky 2013, JMLR 14](https://jmlr.org/papers/v14/johnson13a.html)
- Recurrent switching linear dynamical systems (rSLDS): discrete-state transition probabilities depend on the continuous latent state, so switching is tied to *regions of state space*. Inference uses Pólya-gamma augmentation — [Linderman et al. 2017, AISTATS, arXiv 1610.08466](https://arxiv.org/abs/1610.08466). Tree-structured rSLDS give a hierarchical (coarse-to-fine) partition of state space — [Nassar et al. 2019, ICLR, arXiv 1811.12386](https://arxiv.org/abs/1811.12386). Switching *nonlinear* dynamical systems via collapsed amortized variational inference — [Dong et al. 2020, ICML](https://proceedings.mlr.press/v119/dong20e.html)
- `ssm` (Linderman lab) supports HMMs, autoregressive HMMs, input–output HMMs, hidden semi-Markov models, LDS, SLDS and rSLDS. Its README noted a JAX refactor in progress — [ssm repository](https://github.com/lindermanlab/ssm) (the README contents were seen through a search snippet, not fetched directly). `dynamax` (JAX; JOSS 2025) provides inference and learning for HMMs, linear Gaussian SSMs and nonlinear/generalized Gaussian SSMs; SLDS support was not confirmed — [Dynamax JOSS paper](https://joss.theoj.org/papers/10.21105/joss.07069)
- Hidden Markov trees: exact HMM inference on tree-structured (rather than chain) dependency graphs — [Crouse, Nowak, Baraniuk 1998, IEEE Trans. Signal Process.](https://doi.org/10.1109/78.668544)

#### Contact-state segmentation precedents (assembly / compliant motion)
- An HMM used as a process monitor for robotic assembly: force/torque signals, events = contact-state changes, Baum-Welch training — Hovland & McCarragher, "Hidden Markov models as a process monitor in robotic assembly". The venue is inconsistent across sources: IJRR 17(2), 1998 vs. [Modeling, Identification and Control 20(4), 1999 (DOAJ)](https://doaj.org/article/fb80376d38f54f17a280ebffc1f3040e)
- Contact-state estimation combining multiple-model estimation with an HMM over a task-based contact-state network, illustrated on 3-D peg-in-hole — [Debus, Dupont, Howe, IJRR 2004 (PDF)](https://www.bu.edu/biorobotics/publications/IJRR-04.pdf)
- A particle filter jointly estimates the discrete contact formation and continuous geometric parameters over a contact-state graph, discriminating 245 contact formations in real time — Meeussen, Rutgeerts, Gadeyne, Bruyninckx, De Schutter, IEEE T-RO 23(2):218–231, 2007 (bibliographic details via [DBLP](https://dblp.org/pid/72/2256); related reference list in [Segmenting and Sequencing of Compliant Motions, arXiv 1809.00686](https://arxiv.org/pdf/1809.00686))
- HMMs on F/T data for human peg-in-a-hole skill capture — [Loughborough repository entry](https://repository.lboro.ac.uk/articles/conference_contribution/Human_skill_capture_A_hidden_Markov_model_of_force_and_torque_data_in_peg-in-a-hole_assembly_process/9555617)

### Inferences

**What to segment on (per trajectory; inputs are `traj`, `step` and the per-step fields):**
1. *Jump signal.* Unit friction-corrected wrench ŵ_t, or the unit contact normal. A make/break event or a non-smooth mode switch is a jump. Use the `ruptures` PELT with a "normal" or "rbf" (kernel) cost, or BOCPD with a von Mises–Fisher or Gaussian predictive model.
2. *Kink signal (tangent-continuous boundaries).* The turning rate κ_t = angle(ŵ_t, ŵ_{t+1}) / |q_{t+1} − q_t|, or the residual of a local linear fit of ŵ against arc length. Inside a flat-face mode κ ≈ 0; on a rim/torus mode κ ≈ constant; at a C1 junction κ jumps. Equivalently, run a piecewise-*linear* (continuous) cost or ℓ1 trend filtering on ŵ_t: a kink in ŵ is a change of slope.
3. *Dimension signal.* Over a sliding window, the numerical rank of the twist vectors (local PCA of `twist` in the reduced chart). The walk steps in ker J_A, so the twist span has dimension 4 − |A| in the symmetry-reduced coordinates (3 on single-contact sheets, 2 on two-contact strata). A rank drop is a changepoint. `n_active` already records the active-set size. Arguably it is sampler state rather than a privileged analytic label, but it cannot distinguish two single-contact modes.

The joint per-step feature vector [ŵ, κ, local twist rank] can go into a single multivariate kernel-CPD (KCP or `ruptures` "rbf"). Choose penalties by BIC-like rules or by a stability plot (number of changepoints vs penalty).

**Model-based alternative (CHAMP-style).** Use as the segment model a *local constraint model*: within a segment, twists satisfy n(q)·qdot ≈ 0 with n smoothly varying, and wrench ≈ f_n·n + friction. Score the segment fit as CHAMP does. This turns segmentation into fitting a contact constraint per segment, much as CHAMP fits articulation models.

**Segment-then-cluster.** Each segment gets a descriptor: mean reduced pose, mean ŵ (or normal), mean κ, twist rank, and the fitted local affine map ŵ ≈ A q + b (see §4). Then:
- (a) Build a segment graph. Must-link segments that overlap spatially and have similar descriptors. Cannot-link (or weight 0) consecutive segments in the same trajectory separated by a confident changepoint.
- (b) Cluster with constrained spectral clustering, or HDBSCAN on the descriptors.
- (c) Collapse to a segment-level transition graph and run PCCA+/GPCCA (§1). This reproduces the user's stratum graph with *discovered* nodes in place of analytic modes, which should recover family-level ARI ~0.85 and possibly better.

**Bayesian switching models adapted to this data:**
- *IO-HMM / switching regression.* Hidden mode k_t, emission ŵ_t ~ N(A_k [q_t; v̂_t; 1], Σ_k), transitions along the trajectory. `ssm` supports input–output HMMs. The "dynamics" here is the pose->wrench *map*, not a time-series AR model. That is preferable, because the walk's own velocity is controlled by the sampler (RRT targets) and carries little mode information beyond the tangent subspace.
- *Tree-structured chain.* Because trajectories branch from parent samples, the latent mode sequence lives on the sampler's trajectory tree. Forward–backward generalizes exactly to trees (hidden Markov trees), still linear time. Feasible for millions of samples with fixed K, though not in `ssm` out of the box.
- *rSLDS / tree-rSLDS.* Mode transitions depend on location (rSLDS recurrence = softmax over linear functions of the continuous state, i.e., polyhedral regions). That matches "a mode is a region of pose space". Tree-rSLDS's hierarchical partition matches the family -> mode hierarchy.
- *BP-AR-HMM / HDP-HSMM* give the mode count automatically and share modes across trajectories (behaviours ≈ modes). MCMC was demonstrated on motion-capture-scale data (tens of sequences), so subsample trajectories (e.g., 10^3 trajectories) to learn the mode library, then assign all segments by likelihood. HDP-HSMM duration modelling suppresses chattering at near-tangent boundaries.

**Hook-in-eyelet.** There is no wrench, but each free-space trajectory's local diffusion changes inside a narrow passage. Collision rejections (if the walk records them) rise, and the accepted-step covariance becomes anisotropic and low-rank. Changepoints in rejection rate or in step-covariance rank along trajectories flag passage crossings, and segments between them are mode visits.

**Scalability summary.**

| Method | Cost |
|---|---|
| PELT per trajectory | ~O(T) |
| BOCPD with pruning | ~O(T) |
| KCP | O(T²) per trajectory |
| BP-AR-HMM / HDP-HSMM MCMC | subsample needed |
| rSLDS (Laplace-EM in `ssm`) | ~10^5 steps plausible, not verified |
| Tree HMM with fixed K | O(N·K²) |

Total cost for PELT, BOCPD and KCP stays linear in N because trajectories are short.

### Gaps
- I found no published use of BOCPD/PELT/BP-AR-HMM on *simulated contact-surface random walks* for mode discovery. The closest precedents are force-signal HMMs in assembly and CHAMP for articulation/contact changes.
- Maintenance status not verified for: `pyhsmm` (mattjj; HDP-HSMM reference code, believed unmaintained), `bnpy` (scalable memoized VI for BNP HMMs), the CHAMP ROS package (old ROS), and the `ruptures` release cadence. `dynamax` is current (JOSS 2025), but SLDS support there was not confirmed.
- I did not verify whether `ssm`'s IO-HMM supports Gaussian regression emissions on arbitrary inputs, as opposed to only GLM-HMM categorical outputs.

## 4. Hybrid / piecewise-affine (PWA) system identification and mode-count selection

### Takeaway
Hybrid-system identification treats data as generated by K local (affine) models over a partition of the regressor space. Clustering-based PWA ID (Ferrari-Trecate et al.) clusters *local model parameters*, so it detects slope changes even where the function is continuous. That is exactly what tangent-continuous contact boundaries require. Bemporad's PARC (`pip install pyparc`) is a maintained, scalable, block-coordinate-descent PWA regressor with polyhedral regions. Sparse/ℓ0–ℓ1 and set-membership methods give mode-count selection with guarantees but do not scale to millions of points. Contact-specific hybrid learning (LCS learning, template PWA regression on contact examples) is the robotics frontier.

### Cited Findings
- Survey/tutorial: hybrid ID approaches fall into algebraic (GPCA), Bayesian, clustering-based, bounded-error, and mixed-integer optimization families — [Paoletti, Juloski, Ferrari-Trecate, Vidal 2007, "Identification of hybrid systems: a tutorial" (PDF)](https://www.cis.jhu.edu/~rvidal/publications/ECC07_1285_FI.pdf)
- Clustering-based PWA identification (Ferrari-Trecate, Muselli, Liberati, Morari 2003, Automatica): (1) fit a local affine model on each data point's neighbourhood ("local data sets"); (2) cluster the local parameter vectors (weighted k-means) to assign modes; (3) estimate the region boundaries with linear classifiers (e.g., SVMs). Summarized in the [tutorial above](https://www.cis.jhu.edu/~rvidal/publications/ECC07_1285_FI.pdf)
- Lauer & Bloch, *Hybrid System Identification: Theory and Algorithms for Learning Switching Models* (Springer LNCIS 478, 2019) is a self-contained treatment. Chapters cover classification, exact methods, switched linear/affine estimation, PWA estimation, recursive, state-space and nonlinear identification — [Springer](https://link.springer.com/978-3-030-13091-6)
- PARC (Bemporad, IEEE TAC 68:3194–3209, 2023) fits piecewise-affine predictors over a polyhedral partition for regression (ridge) and classification (softmax) at once. It alternates model fitting with a reassignment of points that balances prediction accuracy against piecewise-linear separability of the clusters, is a block-coordinate descent, and converges in finitely many steps to a local minimum. Python package `pyparc` — [PyPI pyparc](https://pypi.org/project/pyparc/); [arXiv 2103.06189](https://ar5iv.labs.arxiv.org/html/2103.06189)
- Switched linear system ID via sparse optimization: identifying each submodel is posed as an NP-hard ℓ0 problem and relaxed to convex ℓ1 minimization — [Bako, "Identification of switched linear systems via sparse optimization" (HAL)](https://hal.archives-ouvertes.fr/hal-00584246). A tighter persistence-of-excitation analysis — [arXiv 2112.02802](https://arxiv.org/pdf/2112.02802). Set-membership ID of switched systems, equivalent to rank minimization of a matrix affine in the unknowns — [Ozay, Lagoa, Sznaier (Automatica)](https://web.eecs.umich.edu/~necmiye/ols11_sub.html)
- A two-stage switched-linear ID uses dynamic programming to find the switching instants along the time series, then relaxes the combinatorial ℓ0 problem to ℓ1 — [Zheng & Ye 2024, arXiv 2407.02743](https://arxiv.org/abs/2407.02743v1)
- Template-based PWA regression learns linear modes and their validity domains from simulated trajectories. Its example is a double pendulum with soft contacts and nine modes (each joint's contact inactive / active-left / active-right), recovering roughly a 3×3 grid of regions — [arXiv 2305.08686](https://arxiv.org/pdf/2305.08686)
- Learning linear complementarity systems (LCS) from data (Jin, Aydinoglu, Halm, Posa) has been cited as an efficient way to learn PWA systems with tens of thousands of hybrid modes. This characterization is second-hand, from a citing paper — [arXiv 2112.13284](https://arxiv.org/pdf/2112.13284); task-driven hybrid model reduction for dexterous manipulation — [Jin & Posa, arXiv 2211.16657](https://arxiv.org/html/2211.16657)
- Neural Hybrid Automata learn dynamics with multiple modes and stochastic transitions. They note that most prior work targets (linear) PWA systems and that continuous-optimization approaches neglect event stochasticity and mode recovery — [Poli et al. 2021, arXiv 2106.04165](https://arxiv.org/pdf/2106.04165)
- Tree-based robust identification of general hybrid automata from noisy data (HSCC 2025) — [ACM DL](https://dl.acm.org/doi/pdf/10.1145/3716863.3718030)
- Adaptive PWA inverse modelling of hybrid systems (Elhamifar, Burden, Sastry 2014) — [PDF](https://people.eecs.berkeley.edu/~sastry/pubs/Pdfs%20of%202014/ElhamifarAdaptive2014.pdf). Sparse subspace clustering (self-expressive ℓ1 coding of each point by others in its subspace) clusters data lying on a union of subspaces — [Elhamifar & Vidal 2013, TPAMI, arXiv 1203.1005](https://arxiv.org/abs/1203.1005)

### Inferences

**What the "system" is.** The useful formulation is *static* hybrid regression, not dynamic ID:
- Regressor x = reduced pose (rho, radial tilt, |tangential tilt|, depth), optionally plus the twist direction v̂ so friction is explained rather than treated as noise.
- Output y = unit wrench direction ŵ, or the contact normal / friction-corrected normal-wrench direction.
- Within a mode, y(x) is smooth but not affine (curved C-obstacle surfaces). So use *over-segmented* PWA (K well above the true mode count), then merge adjacent pieces whose models agree at the shared boundary. A jump or slope change keeps them apart.

**A second, purely kinematic "system" is a switched linear constraint.** In mode m, twists satisfy J_m(q)·qdot = 0. Locally the twist vectors lie in a union of subspaces (the tangent spaces of the strata present nearby). Sparse subspace clustering or GPCA on twists within a pose neighbourhood separates the modes. The tangent space of a 3-D sheet differs from that of a different sheet even where their *normals* agree to first order, and a 2-contact stratum's 2-D tangent space is a subspace of both adjacent sheets' spaces. This needs only `twist`, not wrench.

**Clustering-based PWA ID (Ferrari-Trecate) maps directly onto the trajectory data.** Define each sample's local data set as its trajectory window (±m steps) plus kNN neighbours in pose. Fit ŵ ≈ A x + b, so θ_i = vec(A, b), and cluster θ. The slope part A (the "shape operator" of the contact surface, in effect) separates tangent-continuous modes: face-on-mouth has A ≈ 0 (constant normal), edge-on-rim has A ≠ 0. Windows that straddle a boundary give outlier θ. Ferrari-Trecate's weighting by local-fit confidence down-weights them, and they mark the boundary. With within-mode curvature, θ drifts smoothly inside a mode, so use a graph-connected clustering of θ (kNN graph in pose, edges gated by θ similarity), or the network lasso of §5, rather than plain k-means.

**PARC.** Fit on a subsample (e.g., 10^5 points) with K = 20–50. It returns a polyhedral partition (softmax classifier) plus affine models; then merge regions.
- Strength: the polyhedral partition generalizes to all millions of samples by one softmax evaluation.
- Weakness: modes in reduced coordinates are not convex polyhedra (annular rim regions, curved bore boundaries). Hence over-segment and merge, using trajectory transitions between PARC regions to decide merges. This is a PCCA+-style coarse-graining of the PARC regions, consistent with §1.

**Trajectory-wise switched ID (Zheng & Ye, Bako).** The DP-for-switching-instants stage is the hybrid-ID analogue of PELT on each trajectory (§3). The ℓ1 stage then fits a small set of shared submodels across segments. That suits mode-count selection by sparsity, but solve it on segment summaries, not raw samples.

**Mode-count selection options:**
- spectral eigengap or PCCA+ crispness (minChi) on a coarse transition graph;
- BNP priors (HDP / beta process) in switching models;
- the PELT penalty with a BIC-style choice;
- PARC K by cross-validated prediction error, then merging;
- the sparsity level in ℓ0/ℓ1 hybrid ID (Bako; Ozay et al.).

A robust practice is to choose K where several criteria agree, and to report the stability of partitions across K (the user's script already scans a range of k).

**Contact-robotics relevance.** LCS learning and template-PWA regression show that contact modes *are* learnable as hybrid structure from simulated trajectories. They require the dynamics (accelerations, forces) to follow a complementarity or PWA template, whereas the user's data are quasi-static contact wrenches under a commanded slide. The pose->wrench map is the right analogue of their "dynamics".

### Gaps
- Scalability of `pyparc` to ≥10^6 points and its handling of multi-output (6-D wrench) targets was not benchmarked in sources I read. Its reported use is on moderate-size regression/control problems.
- I did not verify current availability of MATLAB hybrid-ID toolboxes: the Hybrid Identification Toolbox (HIT, Ferrari-Trecate) and Lauer's code accompanying the book.
- I found no paper applying PWA/hybrid ID to *pose->contact-wrench* maps on a configuration-space contact surface. The closest is template-PWA on a soft-contact pendulum and LCS learning.
- For the Ferrari-Trecate 2003 Automatica paper I cite the tutorial summary, not the original article (not fetched).

## 5. Piecewise-continuous regression and discontinuity-boundary estimation for scattered low-dimensional data (Jump GPs, treed GPs, mixtures of experts, polynomial annihilation, Mumford–Shah / Potts on graphs)

### Takeaway
For the scattered map pose -> wrench direction (or contact normal) in 3–4 symmetry-reduced dimensions, the most directly applicable tools are:
- Jump GPs and their 2025 modular/deep successors (local GPs that partition each neighbourhood across the jump);
- polynomial-annihilation edge detectors, which flag jump *and* derivative-jump locations in irregular data;
- graph-based Potts / fused-lasso / network-lasso estimators on a kNN-plus-trajectory graph, which yield piecewise-constant or piecewise-affine fits whose constant pieces *are* the modes.

The graph estimators scale best, and on the sampler's own trajectory tree the fused lasso reduces to near-linear-time chain problems. Treed GPs fit only where boundaries are roughly axis-aligned (e.g., depth = 0 at the mouth).

### Cited Findings

#### Jump Gaussian processes
- The Jump GP (Park 2022, JMLR) makes a local GP estimate at each test point. The local data are partitioned by a learned function so that points from different continuous pieces are not mixed. The partition function and covariance parameters are optimized jointly by likelihood maximization — [JMLR v23 (21-1472)](https://www.jmlr.org/papers/v23/21-1472.html); [arXiv 2104.06487](https://arxiv.org/pdf/2104.06487). The JMLR site and the author's code page disagree on the article number. Code (MATLAB) is listed on [Chiwoo Park's code page](https://www.chiwoopark.net/code-and-dataset)
- Modular Jump GPs (Flowers, Franck, Binois, Park, Gramacy; arXiv May 2025, rev. Nov 2025) note that joint inference in the JGP "can be fraught with difficulty". They keep two ideas modularly: (1) adaptive neighbourhood sizes that respect the manifold of discontinuity, and (2) a cluster-based latent feature that captures regions of distinct output level on each side of the jump. Each helps on its own and the benefits compound — [arXiv 2505.15557](https://arxiv.org/abs/2505.15557)
- Deep Jump GPs (Xu & Park, 2025) handle high-dimensional inputs. They apply locally linear projections to low-dimensional latent features, with location-specific projection matrices tied by a GP prior, and feed the latent features into a JGP (a two-layer GP/JGP) — [arXiv 2510.21974](https://arxiv.org/html/2510.21974)
- Active learning of piecewise GP surrogates, i.e., sequential design that places samples near jump boundaries — [arXiv 2301.08789](https://arxiv.org/abs/2301.08789)

#### Treed GPs
- Bayesian treed GPs partition the input space by axis-aligned binary trees with independent stationary GPs in the leaves, giving nonstationary fits with discontinuities across leaves; implemented in the R package `tgp` — [Gramacy & Lee 2008, JASA, arXiv 0710.4536](https://arxiv.org/abs/0710.4536)

#### Polynomial-annihilation (PA) edge detection on scattered data
- Archibald, Gelb & Yoon (2005, SIAM J. Numer. Anal. 43(1):259–279) give a local polynomial-annihilation edge detector for irregularly sampled multidimensional data. The detector converges to zero away from discontinuities. A minmod over several orders keeps fast convergence while suppressing oscillations near edges, and the method distinguishes jumps from steep gradients even with sparse non-uniform data — [ORNL publication record](https://www.ornl.gov/publication/polynomial-fitting-edge-detection-irregularly-sampled-signals-and-images)
- Extension to discontinuities in *derivatives* — [Archibald, Gelb, Yoon 2008, "Determining the locations and discontinuities in the derivatives of functions"](https://impact.ornl.gov/en/publications/determining-the-locations-and-discontinuities-in-the-derivatives-/). Detection of gradient discontinuities from irregular samples by differentiating the data first and then applying PA — [Saxena, Gelb, Mittelmann 2009, Commun. Comput. Phys.](https://journal.global-sci.org/intro/article_detail/cicp/7758.html)

#### Potts / ℓ0, fused lasso, trend filtering and network lasso on graphs (discrete Mumford–Shah analogues)
- ℓ0 edge-penalized estimation of piecewise-constant signals on graphs (graph Potts model) is intractable to minimize exactly. Alpha-expansion (graph cuts) reaches an approximate minimizer in polynomial time with the same statistical risk guarantees, which are minimax rate-optimal for low average degree. Effective-resistance edge weights help on irregular graphs. ℓ0 beats the ℓ1/TV relaxation at high SNR — [Fan & Guan 2018, Ann. Statist. (arXiv 1703.01421)](https://arxiv.org/pdf/1703.01421)
- The generalized lasso / graph fused lasso penalizes differences across graph edges; 1-D and 2-D fused lasso are the chain and grid cases — [Tibshirani & Taylor, arXiv 1005.1971](https://arxiv.org/abs/1005.1971). Graph trend filtering gives higher-order (piecewise-polynomial) analogues — [Wang, Sharpnack, Smola, Tibshirani 2016, JMLR (arXiv 1410.7690)](https://arxiv.org/abs/1410.7690)
- A DFS-based chain surrogate runs the fused lasso on general graphs in linear time and matches the optimal rate on trees — [Madrid Padilla et al., seminar abstract](https://pi.math.cornell.edu/m/node/8473). Sparsistency conditions for the edge lasso — [Sharpnack et al. 2012, AISTATS](https://proceedings.mlr.press/v22/sharpnack12.html). Selective inference (p-values) for segments found by the graph fused lasso — [Chen, Jewell, Witten, arXiv 2109.10451](https://arxiv.org/abs/2109.10451v2)
- Network lasso: each node has its own model x_i, with a sum-of-norms penalty λ Σ w_jk ||x_j − x_k||_2 over edges. This fuses neighbouring nodes into clusters that share a model (simultaneous clustering and optimization), solved by ADMM — [Hallac, Leskovec, Boyd 2015, KDD, arXiv 1507.00280](https://arxiv.org/abs/1507.00280)
- Pottslab (MATLAB) implements Potts / piecewise-constant partitioning of vector-valued signals and images — [Storath & Weinmann, Pottslab on GitHub](https://github.com/mstorath/Pottslab)

### Inferences

**The function to regress.** Raw wrench mixes normal force and sliding friction. Friction depends on slip direction, which changes along trajectories, so it blurs the pose->wrench map. Options:
- (i) Use the simulator's per-contact unit normal (the `contacts` field) where n_active = 1. It is a measurement, not an analytic mode label, though the user may class it as privileged.
- (ii) Regress ŵ on (pose, v̂), so friction is part of the model.
- (iii) Correct for friction: the constraint (normal) wrench is reciprocal to the admissible twist (zero power, W_n·V = 0), while sliding friction does negative power. Remove the component of w along the power-pairing direction of V. For a single contact this leaves an estimate of the normal-wrench direction up to the residual tangential friction terms.

The target is then ŵ ∈ S^5 (or S^2 for a contact normal) over x ∈ R^4 lying on 3-D sheets and 2-D strata.

**Manifold caveat for all scattered-data methods.** The samples are not full-dimensional in R^4: single-contact samples lie on 3-D sheets and two-contact samples on 2-D strata. Ambient polynomial stencils (PA) and isotropic GP kernels are ill-conditioned across the normal direction. Work in local tangent charts: local PCA of kNN poses, or the twist span from the trajectory. Points on 2-D strata are themselves *on* the boundary between sheets, so treat them as part of the edge set, not as a separate cluster to be found by regression.

**Recipe A (JGP / MJGP as edge-aware affinity).**
1. For each sample i (or a 10^5 subsample), fit a JGP/MJGP on its kNN neighbourhood (along-trajectory plus pose neighbours).
2. The fitted local partition says which neighbours lie on i's side of the jump. Set w_ij = 1 for same side and 0 otherwise.
3. Connected components or spectral clustering on this graph give the modes.

MJGP's "cluster-based latent feature" and adaptive neighbourhoods are designed exactly for neighbourhoods that straddle a discontinuity manifold. Cost is O(N·n_loc^3) with n_loc ≈ 50–150 and embarrassingly parallel per point, so it is feasible on a subsample but heavy for millions in MATLAB. JGP targets *jumps*; tangent-continuous boundaries need the derivative-aware variants below.

**Recipe B (PA edge indicator -> flood fill).**
1. In local 3-D tangent charts, compute the PA jump indicator for each component of ŵ.
2. Compute it also for the gradient of ŵ (Saxena–Gelb–Mittelmann: differentiate, then detect). This catches C1 tangent-continuous boundaries.
3. Threshold with minmod over orders.
4. Delete edge-flagged samples (and the 2-D-strata samples) from the kNN-plus-trajectory graph.
5. Take connected components, then reassign the deleted samples by nearest component along their trajectory (core-set logic, §1).

This is linear in N given a kNN index. Few-dimensional data is exactly where PA was designed to work.

**Recipe C (graph Potts / fused lasso / network lasso).** Build G with:
- the sampler's own edges (trajectory steps and parent->child links), which form a *tree* per worker and are guaranteed continuous motions;
- kNN pose edges in the reduced chart.

Then choose a penalty:
- *Piecewise-constant:* graph Potts (ℓ0) on ŵ via alpha-expansion, or ℓ1 fused lasso. Pieces = candidate modes (jump boundaries).
- *Piecewise-affine (catches slope changes):* network lasso with per-node local affine models x_i = (A_i, b_i), loss Σ_i ||ŵ_i − A_i q_i − b_i||^2 plus a small ridge, and penalty Σ_edges w_jk ||(A_j, b_j) − (A_k, b_k)||. Fused nodes share an affine model. This is the convex, graph-native form of clustering-based PWA ID (§4) and needs no K.
- *On the trajectory tree alone,* the fused lasso / 1-D Potts is solvable in near-linear time (DFS chain surrogate). That is a cheap first pass over millions of samples, followed by kNN edges to merge pieces across trajectories.
- *Significance:* selective inference for the graph fused lasso can test whether a detected boundary is real, which helps separate true mode boundaries from noise-induced breaks.

**Treed GP.** Only for sub-regions with near axis-aligned boundaries: the mouth plane (depth = 0), the bore wall (rho ≈ clearance), and tilt-sign boundaries. `tgp` is cubic per leaf, so subsample to ~10^4.

**Mixtures of experts / PARC.** A gating network over pose (softmax gives polyhedral regions, as in PARC and the rSLDS recurrence) with local regressors of ŵ. The hard-EM version is PARC (§4). SGD-trained neural MoE scales to millions but inherits local minima and needs K.

**Hook-in-eyelet.** No wrench, so no natural discontinuous function. Possible continuous "functions":
- *Clearance (signed distance to the eyelet, via `mj_geomDistance`).* Its sublevel-set topology / watershed separates regions at narrow passages (minima of clearance along passages). This is a Morse/watershed construction rather than jump regression.
- *Visibility / reachability indicators.* Discontinuous in pose; computable along trajectories.

The MSM route (§1) is more natural there.

#### Integrated pipeline sketch for this repo (synthesis across §1–§5; my inference)
1. **Per-sample features.** Reduced pose x; friction-corrected unit wrench ŵ (or contact normal); twist direction v̂; local tangent chart and twist rank from a ±m-step trajectory window.
2. **Per-trajectory changepoints.** PELT (`ruptures`) on [ŵ, κ = turning rate, twist rank], linear time overall. This yields segments and cuts trajectory edges at changepoints.
3. **Graph.** Trajectory-tree edges (minus cut edges) plus kNN pose edges gated by ŵ and local-affine-model similarity. Optionally fit a network lasso / graph Potts on this graph for piecewise-affine pieces.
4. **Microstates.** Segments, or Potts pieces, or k-means in [x, λŵ]. Build a directed count matrix from trajectory steps and parent links.
5. **Coarse-graining.** PCCA+ (symmetrized counts) or GPCCA (directed counts); choose k by crispness / eigengap / implied-timescale stability. Optionally, a VAMPnet or TCL classifier on (x, ŵ, v̂) as a parametric pose->mode map for new data.
6. **Validation.** ARI/NMI against the analytic modes (as `contact_graph_spectral.py` does), separately for jump-type and tangent-continuous boundaries. Selective-inference p-values for boundaries. Stability across penalties and k.
7. **Hook-in-eyelet.** Skip 1–3; use pose microstates, then MSM/PCCA+/GPCCA, plus TPT committors for passages, plus Laplacian-representation learning or a VAMPnet for a parametric classifier.

### Gaps
- No code release was found for Modular JGP or Deep JGP. The original JGP code is MATLAB, and I could not verify a maintained Python implementation.
- I found no published comparison of JGP vs. PA vs. graph Potts on data lying on a lower-dimensional manifold *within* the input space, which is the stratified-sample situation here. The manifold caveat above is my inference.
- Mixture-of-GP-experts references (e.g., Rasmussen & Ghahramani's infinite mixture of GP experts) and Mumford–Shah formulations on point clouds / graphs (beyond Potts / fused lasso) were not fetched or verified in this session.
- I found no source that quantifies how sliding friction blurs pose->wrench discontinuities, or validates the reciprocity-based friction removal sketched above. The user can check this empirically using the `contacts` normals as reference.
