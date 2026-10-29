# Separating contact modes needs physics, not geometry

Published methods cover five of the six failures this repo has hit: mixed-dimension strata, creases, density variation, choosing the number of clusters, and narrow passages. None separates the **tangent-continuous** mode boundaries, which have no signature in pose space at all. Only physical signals can separate those: which feature pair touches (recoverable from the wrench's line of action), the local pose-to-wrench model, or per-feature-pair constraint functions. The repo's own numbers sharpen the priorities. A supervised classifier on **pose alone reaches 0.987 balanced accuracy on families**, so the gap to unsupervised ARI ~0.3 (pose) and ~0.85 (with wrench) comes from the clustering objective and graph construction, not from missing information. At the exact-mode level, even the privileged friction-free contact wrench only lifts the ceiling to **0.761**. Finer modes therefore need either a codimension-aware label definition or features the repo has not yet built. The changes with the most leverage are structural:

- Replace balanced-cut spectral clustering with **k-free signed partitioning** (multicut, mutex watershed), with repulsive edges from local physical jump tests.
- Estimate stratum dimension per sample in three independent ways, and cluster within dimension layers.
- Detect boundaries as 1-D changepoints along each random walk instead of in 4-D scatter.
- For the hook, treat modes as kinetic and topological objects: Markov-state coarse-graining, merge trees of C-space clearance, and a Gauss-linking-integral threading coordinate.

No published work clusters sampled contact or free-space configurations and scores the result against physical mode labels, so every application below is an untested inference. The report marks the difference throughout. It ends with ten experiments ordered by expected gain per day. The first takes about a day: it breaks down the existing supervised ceiling to show how much exact-mode headroom unsupervised work can reach at all.

## The repo's ceilings show the bottleneck is the objective, not the information

**How to read this report.** A sentence with a citation reports a published result. A sentence or table cell marked **Inference** extends those results to this repo and has not been tested. Nothing in the research applies any method below to sampled contact-mode or C-space free-space data and scores it against physical labels. The closest unsupervised precedents cluster tactile signals into contact events ([Kroemer & Su, ICRA 2018](https://www.ri.cmu.edu/app/uploads/2019/03/Kroemer_Su_ICRA_2018.pdf)), or judge C-space regions only by coverage, region count and runtime ([Werner et al., VCC](https://arxiv.org/html/2310.02875v2)). On this point the repo's benchmark harness is ahead of the literature.

The right frame comes from a 2026 paper whose title states its thesis: **contact modes are strata**. Each mode's configuration set S_A has dimension n − |A|, because "every contact that holds removes one direction". Strata are found by motion rather than by enumeration, and the authors note that active-set strata need not be connected ([Kyaw & Kelly, arXiv 2608.15541](https://arxiv.org/html/2608.15541)).

In the repo's symmetry-reduced 4-D chart this gives:

- 3-D single-contact sheets;
- 2-D two-contact strata;
- lower-dimensional corners;
- for the hook, free-space modes, which are the connected pieces of the empty-contact stratum.

Mode boundaries come in three kinds:

- **Dimension change:** a sheet meets a two-contact stratum.
- **Angle change:** two sheets meet at a crease, and the normal jumps.
- **Identity change:** two sheets meet tangentially, so only *which feature pair* touches changes.

The six failures are these three kinds plus three cross-cutting problems: non-uniform density, choosing the number of clusters, and narrow passages.

### What the separability ceiling shows

The repo's `mode_discovery_bench.py separability` already measures the information ceiling. It trains a gradient-boosted classifier with spatial-block cross-validation on 200,000 balanced samples (`run_20261024_221603/discovery/balanced200000/separability.json`).

| Feature set | Family (balanced accuracy) | Mode (balanced accuracy) |
|---|---|---|
| pose | 0.987 | 0.617 |
| pose + wrench | 0.988 | 0.711 |
| pose + wrench + normal + curvature | 0.991 | 0.748 |
| pose + contact wrench (privileged, friction-free) | 0.991 | 0.761 |

**Inference.** Two conclusions follow:

1. **Families are a function of pose.** Balanced accuracy and ARI are different scales, but a four-family labelling that is 99% correct would score ARI well above 0.9. Unsupervised pose-only ARI of ~0.3 therefore reflects a failure of the partition objective and edge construction, not missing information. Wrench features help the unsupervised methods mainly because they turn mode boundaries into large *local* distances that a graph method can see; the supervised pose-only result shows they add little information.
2. **Better wrench measurement will not close most of the exact-mode gap.** The privileged friction-free contact wrench adds only 0.013 over the measured wrench, so most of the remaining 24% exact-mode error has another source. Balanced accuracy weights every mode with at least 200 samples equally, so thin two-contact modes with poor recall pull the score down. Those modes are defined by a tolerance band around a measure-zero crease, so part of the ceiling is label definition rather than separability. The repo's `codim_aware_scores.py`, which lets a two-contact mode share a cluster with its constituent single-contact modes, is the right lens for this.

### Why spectral clustering stalls here

Normalized-cut spectral clustering has an exact random-walk reading. Minimizing the normalized cut finds sets that the walk P = D⁻¹W rarely leaves ([Meilă & Shi, AISTATS 2001](https://proceedings.mlr.press/r3/meila01a.html)). On the peg's contact surface most mode boundaries are not bottlenecks: the walk slides from one sheet to the next with pose continuous. The leading eigenvectors therefore encode the large-scale geometry of the surface, not the modes.

**Inference.** Effective resistance is also a random-walk (commute-time) quantity, which explains why the repo's effective-resistance cuts found no signal.

Near intersections and sharp edges the problem gets worse. There the graph Laplacian does not converge to the Laplace–Beltrami operator but to a different first-order operator, and the few points near singularities contribute disproportionately ([Belkin, Que, Wang & Zhou, COLT 2012](https://proceedings.mlr.press/v23/belkin12.html)).

The benchmark shows how fragile this is. In its cached k = 4 runs, `spectral-normal` and `spectral-wrench` reach **ARI ≈ 0.00 and 0.01**. For `spectral-normal`, a single cluster recalls at least 99% of every family. A tuned wrench block in `knn_pose_spectral.py` reaches ~0.85 on the same problem.

### The normal feature is already the standard tangent signal

**Inference.** The repo's "normal" feature, the upper triangle of the sign-free outer product nnᵀ, is already the textbook tangent signal. Three published constructions reduce to it for codimension-1 sheets in R⁴:

- a 2026 paper lifts each point to (position, tangent projector) with metric d² = ‖p−q‖² + (α/2)‖P_U − P_V‖²_F ([Petrov et al., "Tangent Blow-Ups"](https://arxiv.org/html/2605.18215)), and ‖P_U − P_V‖ equals ‖nnᵀ − mmᵀ‖ in this setting;
- the projection term in local-PCA spectral clustering ([Arias-Castro, Lerman & Zhang, JMLR 2017](https://jmlr.org/papers/v18/14-318.html)) is the same quantity;
- the principal-angle affinity of SMMC ([Wang et al., TNN 2011](https://www.lamda.nju.edu.cn/publication/tnn11smmc.pdf)) is a monotone function of it.

Swapping one tangent-affinity method for another will therefore plateau. Gains have to come from three places: better tangents at creases, a different partition objective, or new physical signals.

| Failure | The boundary signal that actually exists | Best-supported technique families | Strongest published evidence | Status for this repo |
|---|---|---|---|---|
| Mixed-dimension strata | Local dimension drops from 3 to 2; normal-cone rank rises from 1 to 2 | Hidalgo / TwoNN / GRIDE; tensor voting; Voronoi covariance measure (VCM); wrench-cone and twist-span rank | Hidalgo separates intersecting manifolds whose dimensions differ by 1, NMI 0.84–0.89 | **Inference:** works only if two-contact strata carry their own samples |
| Creases | Normal jump with dihedral angle Θ | Local-PCA spectral clustering, LAPD, Deutsch–Medioni junction removal, tangent blow-up, HADES | LAPD between-cluster distance ≳ Θ; tangent methods need > π/4 | Normal feature already equals the first-order lift; tangent quality is the lever |
| Tangent-continuous boundaries | Contact identity (which feature pair), curvature | Wrench line of action; local-model clustering; multi-model fitting; per-feature-pair implicit functions; kink changepoints | None on C-space data; contact sign vectors are identical on both sides | **Inference** throughout; the main source of exact-mode headroom |
| Density variation | None: density reflects the sampler | Density-insensitive estimators, density equalization, per-edge distribution tests | TwoNN is insensitive to density | **Inference:** density-driven methods will cut at sampling changes |
| Choosing K | Sign of edge evidence, persistence, posterior | Multicut / mutex watershed / GASP; LLPD eigengap; LAPD persistence; PCCA+ crispness; label costs | Multicut sets both the assignment and the number of clusters | **Inference:** edge costs must be calibrated without labels |
| Narrow passages / threading | Flow bottlenecks; homotopy class | PCCA+ / GPCCA / transition path theory; clearance merge trees; VCC cliques; orientation slicing; linking integral | Laplacian eigenvectors find rooms and doorways | **Inference:** threading is not always a clearance bottleneck |

## Dimension drops are visible only where two-contact strata hold samples

### Hidalgo is the best-evidenced dimension segmenter

Among published methods, Hidalgo has the best evidence for separating samples by intrinsic dimension. It models the ratio of second- to first-neighbour distances, whose law depends only on local dimension (the TwoNN principle), as a Bayesian mixture of Pareto laws. A neighbourhood-uniformity prior then ties each point's label to its q nearest neighbours.

Reported results ([Allegra et al., Sci. Rep. 2020](https://publicatt.unicatt.it/retrieve/e309db6f-8864-0599-e053-3705fe0a55db/SCIREP_Hidalgo.pdf)):

- Five *intersecting* Gaussians of dimension 1, 2, 4, 5 and 9: **NMI 0.89**.
- Five curved, intersecting manifolds including a Swiss roll and a 3-torus: **0.84**.
- K is chosen by stopping when the average log-posterior stops improving.
- Without the neighbourhood prior, the method "is not able to correctly separate the points" (NMI < 0.001). The recommended setting is q = 3, ξ = 0.8.
- It is insensitive to density because it uses only first- and second-neighbour distances.

Comparators fared worse in the same study. SMCE "cannot correctly retrieve intersecting manifolds", and a smoothed local-dimension baseline merged the 4-D and 5-D manifolds.

Costs and code:

- In practice Hidalgo is **roughly quadratic in N**, so it needs subsamples of 1e4–1e5 points.
- Code is in the original repository ([micheleallegra/Hidalgo](https://github.com/micheleallegra/Hidalgo); MATLAB and Python, no releases) and in the R package intRinsic ([Denti et al.](https://arxiv.org/pdf/2102.11425)).
- DADApy, the maintained Python library, provides TwoNN, GRIDE and density-peak clustering but does not list Hidalgo ([DADApy](https://github.com/sissa-data-science/DADApy)).
- scikit-dimension exposes GRIDE, TwoNN, local PCA and a dozen other estimators ([skdim API](https://scikit-dimension.readthedocs.io/en/latest/api.html)).

### The deciding condition is sampling mass

The original stratification-learning paper notes that strata of different dimension need a mixture-model sampling assumption, "or else lower strata get no points" ([Bendich, Wang & Mukherjee, SODA 2012](https://sci.utah.edu/~beiwang/publications/Stratification_Learning_BeiWang_2012.pdf)).

**Inference.** The repo's sampler steps inside ker J_A and projects onto active sets, so two-contact strata probably carry their own sample mass. If instead most "two-contact" labels come from sheet samples that fall within `SAMPLE_TOL` of a crease, every dimension estimator will return 3 there. A cheap check decides this: run TwoNN on the analytic two-contact samples alone, and expect about 2.

A second trap is specific to random-walk data. A sample's first and second neighbours are often the previous and next steps of the same walk, which biases TwoNN toward 1. Exclude same-trajectory neighbours, or use GRIDE at higher neighbour orders.

### Geometric routes to per-point dimension

N-dimensional tensor voting gives a reliable *per-point* local dimension and is robust to "varying density, outliers, perturbation and intersections" ([Mordohai & Medioni, JMLR 2010](https://jmlr.org/papers/v11/mordohai10a.html)). Its "ball" eigenvalue peaks at junctions, and Deutsch and Medioni found its tangents near intersections more accurate than local PCA's ([IJCAI 2015](https://www.ijcai.org/Proceedings/15/Papers/485.pdf)). No maintained Python implementation turned up.

The Voronoi covariance measure has a Monte-Carlo version "applicable in any dimension" ([Mérigot, Ovsjanikov & Guibas, TVCG 2011](https://www.lix.polytechnique.fr/~maks/papers/voronoi.pdf)), and a distance-to-measure variant that resists outliers ([Cuel et al., SIAM J. Imaging Sci. 2015](https://hal.archives-ouvertes.fr/hal-01058145)).

**Inference.**

- A 4-D voted tensor is only 4×4, so tensor voting over 1e6 points is a single linear pass.
- The rank of the VCM's normal-cone estimate equals the number of independent active contacts: 1 on a sheet, 2 on a two-contact stratum.

### Physical routes to per-point dimension

The physical analogues are more direct. Each contact contributes one constraint wrench, and the feasible twists form a polyhedral convex cone ([Lynch & Park, *Modern Robotics* §12.1.3](https://modernrobotics.northwestern.edu/nu-gm-book-resource/12-1-3-multiple-contacts)). ECoMaNN picks a manifold's codimension from the eigengap of local covariances, but assumes "the intrinsic dimensionality … remains constant" ([Sutanto et al., CoRL 2020](https://ar5iv.arxiv.org/html/2009.11852)). That assumption is exactly what fails on a stratified set.

**Inference.** Three per-sample codimension estimates are available, and they fail in different ways:

1. **Pose eigengap:** local-PCA eigengap of the sample poses.
2. **Wrench-cone rank:** the numerical rank of friction-compensated generalized forces τ = Jᵀw over a neighbourhood on the same stratum. Expect |A|, with friction blurring it by roughly the friction-cone half-angle.
3. **Twist-span rank:** the walk moves in ker J_A, so the rank of the twist span over a ±m-step window should be 4 − |A|.

Agreement between them is a label-free self-check. The principal-contact structure behind automatically generated contact-formation graphs ([Xiao & Ji, IJRR 2001](https://webpages.charlotte.edu/xiao/pdf/IJRRXiaoJi.pdf)) adds a structural test: a two-contact cluster should border exactly the clusters of its two constituent contacts.

### Topological detectors serve only as sanity checks

Topological singularity detectors are well developed but play a supporting role here:

- **GAD** classifies points by counting long bars in annulus persistent homology. Its authors' implementation is MATLAB plus Ripser ([Stolz et al., PNAS 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7443892)). Per the HADES authors, citing a private communication, it "took at least several hours on parallel processing" ([HADES paper](https://arxiv.org/html/2311.04171)).
- **TARDIS** adds a multiscale Euclidicity score and a persistent intrinsic dimension (BSD-3 Python, described by its authors as a work in progress) ([von Rohrscheidt & Rieck, ICML 2023](https://proceedings.mlr.press/v202/von-rohrscheidt23a.html)).
- **Minimax theory** now exists for recovering strata of different dimensions, including layer count, dimensions, point assignment and tangent spaces, but has no code ([Aamari & Berenfeld, 2024](https://arxiv.org/abs/2405.20066)).
- **Recovery near strata boundaries** from samples "may not be feasible" ([Mileyko](https://arxiv.org/abs/1811.09033)).

**Inference.** A crease where exactly two sheets meet is still a topological 3-manifold, so local homology cannot see it. These detectors fire only at:

- junctions of three or more sheets;
- sheet ends;
- transverse crossings.

They are useful on about 1e4 landmarks as a check, not as the main crease detector. The repo's `codim_aware_scores.py` already encodes "separate by dimension, forgive closure" in scoring. A dimension layer computed before clustering would bring the same idea into the clustering itself.

## Every crease detector is a normal-jump detector, so tangent quality decides

### The tangent-aware spectral family

Spectral clustering on local PCA is the canonical method. It weights edges by exp(−‖y_i−y_j‖²/ε²)·exp(−‖Q_i−Q_j‖²/η²), where Q is the local tangent projector, and assumes:

- K and the dimension d are known;
- all surfaces share d;
- intersections meet at a positive angle.

Its guarantees cover only the simpler threshold and connected-component variants. Of the spectral version, the authors concede that "intersection points may form their own cluster" ([Arias-Castro, Lerman & Zhang](https://ar5iv.labs.arxiv.org/html/1301.2007)).

**Inference.** That "flaw" is a feature here. A crease *is* a two-contact mode, so running the spectral variant with K above the number of sheets may produce two-contact clusters for free.

Theory now covers intersecting manifolds of *different* dimension, with annular proximity graphs under angle constraints as an example construction ([García Trillos, He & Li, JMLR 2023](https://arxiv.org/abs/2107.13610)). The practical limit is empirical. Earlier tangent-affinity methods "only provide satisfactory results when the angle between the tangent planes is large (typically > π/4)" ([Deutsch & Medioni, IJCAI 2015](https://www.ijcai.org/Proceedings/15/Papers/485.pdf)).

### Two published fixes change the objective

**LAPD.** LAPD replaces summed affinities with a minimax path distance: a path between simplices costs its *sharpest turn* ([Chen, Little & Narayan, 2025](https://ar5iv.labs.arxiv.org/html/2507.10710)).

- Within-manifold distance is ≲ τ/e and between-manifold distance ≳ Θ, for two linear manifolds.
- K is picked by persistence.
- Runtime is quasi-linear in n but exponential in d, "impractical when d > 5".
- Runtime experiments reach 35,000 points. Code: [HYfromLA/LAPD](https://github.com/HYfromLA/LAPD).

**Inference.** LAPD's model, d-manifolds meeting along (d−1)-dimensional sets, matches 3-D sheets meeting along 2-D creases. Three caveats apply:

1. The theory covers crossing flats, not folded half-sheets.
2. Samples lying *on* the crease form flat simplices, which the shape filter discards, so they must be peeled off first by the dimension layer.
3. The ~1e6 samples must be thinned to about 3e4.

By contrast, LLPD, the longest-leg path distance on which LAPD builds, separates clusters joined by low-density bridges and carries eigengap guarantees for K ([Little, Maggioni & Murphy, JMLR 2020](https://jmlr.org/papers/v21/18-085.html)). **Inference:** with no density gap at a crease, LLPD would merge the sheets.

**Deutsch–Medioni.** Their pipeline:

1. Detect junction areas from peaks of the tensor-voting ball saliency, and remove them.
2. Spectrally cluster the smooth remainder.
3. Greedily relabel the junction points by minimizing the total variation of tangent spaces, at cost O(j·N·k log k) ([IJCAI 2015](https://www.ijcai.org/Proceedings/15/Papers/485.pdf)).

### Tangent quality near creases is the real lever

The tangent blow-up authors name the failure the repo has seen: near singularities, PCA "averages across intersecting components", and normal noise hurts more than positional noise. Their theorem guarantees a lifted separation of at least (1/√2)·√α·δ for a tangent jump δ ([Petrov et al., 2026](https://arxiv.org/html/2605.18215)). The margin is only as good as the tangents.

Robust estimators exist, but mostly in 3-D code:

- Hough normal voting has O(n log n) cost and a header-only C++ implementation for 3-D ([Boulch & Marlet, SGP 2012](https://imagine.enpc.fr/~marletr/publi/SGP-2012-Boulch-Marlet.pdf)).
- Gauss-map clustering of neighbour normals is dimension-free in concept ([Weber, Hahmann & Hagen](https://drops.dagstuhl.de/entities/document/10.4230/OASIcs.VLUDS.2010.90)).
- VCM and tensor voting are covered in the previous section.

### The measured wrench may be the best crease normal

**Inference.** For frictionless contact, the contact force is parallel to the constraint gradient; with friction it stays inside the friction cone. The measured wrench direction is therefore a normal estimate that does not average across sheets.

The compliant-motion literature supplies the second half of this idea. De Schutter and colleagues used **the reciprocity between ideal contact forces and motion freedoms as the Kalman-filter measurement equation** ([De Schutter et al., IJRR 1999](https://researchportal.sckcen.be/en/publications/estimating-first-order-geometric-parameters-and-monitoring-contac/)).

**Inference.** The edge residual r_ij = |τ̂_i · Δq̂_ij| follows from this. It:

- is a power pairing, so it does not depend on the pose metric;
- is directional;
- fires when a displacement leaves sample i's stratum, even where the wrench magnitude varies smoothly.

It is an edge weight that no clustering the repo has run so far uses.

### The tilt lever L sets every crease angle

SE(3) has no bi-invariant metric ([arXiv 1612.04631](https://arxiv.org/pdf/1612.04631)), so the repo's d² = dρ² + d_depth² + L²(dθ_r² + dθ_t²) must pick a length scale L.

**Inference.** Every dihedral angle, and so every tangent-affinity and LAPD margin, depends on L. Two cheap steps follow:

1. Tabulate the analytic dihedral angle of each peg crease under candidate values of L, and predict which boundaries clear the ~π/4 threshold.
2. Choose L to maximize those angles. Using the analytic model only to set a metric is borderline, so the physical lever arm to the contacting feature is the purer choice.

### Cheaper detectors, and what to skip

HADES is the scalable crease detector. It tests a PCA-projected neighbourhood against the uniform disk with a kernel MMD at O(k²D) per point ([Lim, Oberhauser & Nanda](https://arxiv.org/html/2311.04171)).

- It found the cyclo-octane intersection rings in 24-D in 5 seconds.
- Its theorem covers only transverse intersections.
- No public code was found, but it is about 50 lines of kNN, PCA and MMD.

**Inference.** Its uniform-disk null will flag density gradients as anomalies, so equalize density first.

Two other families are low priority here:

- **Self-expressive subspace clustering.** SSC is provably correct for unions of *linear* subspaces, even intersecting ones of different dimension ([Soltanolkotabi & Candès, Ann. Statist. 2012](https://arxiv.org/abs/1112.4258v5)), and SSC-OMP/EnSC handle 0.5 million points in Python ([ChongYou/subspace-clustering](https://github.com/ChongYou/subspace-clustering)). Deep self-expressive variants have ill-posed objectives whose reported gains came largely from post-processing ([Haeffele, You & Vidal, ICLR 2021](https://arxiv.org/abs/2010.03697v2)). **Inference:** curved codimension-1 sheets leave almost no margin for subspace-preserving codes, so SSC belongs only in the joint pose + wrench space, if anywhere.
- **Connection-Laplacian methods** such as vector diffusion maps ([Singer & Wu](https://arxiv.org/pdf/1102.0075)). **Inference:** in codimension 1 they reduce to the normal agreement the repo already uses. Their only new contribution would be transport *frustration* around cycles, used as an edge-level crease score.

## Tangent-continuous boundaries are visible only through which feature touches

### Why every geometric method is blind here

In the contact-mode enumeration literature, a mode is the sign vector of the twist against the contact-constraint hyperplanes ([Huang, Cheng & Mason, WAFR 2020](https://par.nsf.gov/servlets/purl/10338804)).

**Inference.** Where two single-contact sheets meet tangentially (for example, an edge sliding round a rim with its normal turning smoothly), both sides have the same sign vector, "one contact maintained". Only the *identity* of the touching feature pair changes. Every geometric criterion goes blind at such a boundary:

- the dihedral angle Θ is 0, so the LAPD margin vanishes;
- the first-order tangent lift sees one smooth sheet;
- normal-cone rank stays 1.

Robotics has already met this failure. A Dirichlet-process mixture on end-effector states over-split modes, and its authors warned that "not all mode transitions are accompanied by impacting contacts" ([Khader et al., RA-L 2020](https://arxiv.org/html/1909.04915v3)). Four kinds of signal survive.

### Signal 1: the wrench's line of action locates the contact

Classical intrinsic contact sensing computes the contact location on a known body from a wrist force/torque measurement ([IJRR 1993](https://www.centropiaggio.unipi.it/sites/default/files/sensor-IJRR93.pdf)). The contact particle filter localizes *multiple simultaneous* external contacts by searching for contact points that explain the measured joint torques ([Manuelli & Tedrake, IROS 2016](https://groups.csail.mit.edu/robotics-center/public_papers/Manuelli16.pdf)). The force-to-pose map in peg insertion is not one-to-one ([Jin et al.](https://arxiv.org/pdf/2101.12467)), which is the formal reason pose and wrench together beat either alone.

**Inference.** The peg's geometry belongs to the robot, so using it is not privileged. The pipeline:

1. Recover the line {p : m = p × f} from the measured wrench.
2. Intersect it with the peg surface to find the peg feature: tip edge, shoulder edge, side or face.
3. Map that contact point into the hole frame to find the hole feature.

Properties:

- **Robust to friction**, because the force direction tilts inside the cone but the line still passes through the contact point.
- **Detects two contacts:** a single-line fit leaves a residual couple.
- **Caveats:** the repo's torque normalization (torque about the tip ÷ 12.5 mm) must be undone, and line or patch contacts at zero tilt make the line degenerate.

This turns contact *identity* into a coordinate. It is the most direct candidate for separating tangent-continuous modes, and it is cheap to test inside the existing `separability` harness.

### Signal 2: local models and curvature

Clustering-based piecewise-affine identification fits a local affine model in each neighbourhood and clusters the *parameter vectors*. It uses confidence weights to discount unreliable fits that straddle boundaries ([Ferrari-Trecate et al., via Paoletti et al. tutorial](https://www.cis.jhu.edu/~rvidal/publications/ECC07_1285_FI.pdf)). Several tools follow the same idea:

- **Network lasso** fuses per-node models across graph edges, so clustering and fitting happen together with no K ([Hallac, Leskovec & Boyd, KDD 2015](https://arxiv.org/abs/1507.00280)).
- **PARC** fits piecewise-affine models over a polyhedral partition (`pip install pyparc`) ([PyPI](https://pypi.org/project/pyparc/)).
- **Jump GPs** partition each local neighbourhood across a discontinuity ([Park, JMLR 2022](https://www.jmlr.org/papers/v23/21-1472.html)), and the 2025 modular version adds adaptive neighbourhoods ([Flowers et al.](https://arxiv.org/abs/2505.15557)).
- **Polynomial-annihilation detectors** locate discontinuities in *derivatives* from irregular samples ([Archibald, Gelb & Yoon](https://impact.ornl.gov/en/publications/determining-the-locations-and-discontinuities-in-the-derivatives-/)).
- **ℓ1 trend filtering** places kinks adaptively ([Kim et al., SIAM Review 2009](https://doi.org/10.1137/070690274)).

**Inference.** Regress the friction-corrected wrench direction ŵ on (pose, walk direction). The slope then separates modes that share a normal at their boundary: a face on the mouth gives A ≈ 0, an edge on a rim gives A ≠ 0. This is the physical version of the tangent blow-up's second-order (curvature) lift. The repo's quadric fits target the same signal but suffer the same neighbourhood mixing. A network lasso on the kNN + trajectory graph is the k-free form of that idea.

### Signal 3: one constraint function per feature pair

Multi-model fitting chooses the number of models without a preset K:

- T-Linkage stops agglomeration with outlier rejection ([Magri & Fusiello, CVPR 2014](https://openaccess.thecvf.com/content_cvpr_2014/html/Magri_T-Linkage_A_Continuous_2014_CVPR_paper.html)).
- MultiLink runs model selection inside the linkage rule ([Magri et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Magri_MultiLink_Multi-Class_Structure_Recovery_via_Agglomerative_Clustering_and_Model_Selection_CVPR_2021_paper.html)).
- PEARL adds label costs to a spatially regularized energy ([Isack & Boykov, IJCV 2012](https://www.csd.uwo.ca/~yboykov/Papers/ijcv10_pearl.pdf)).
- Multi-class T-Linkage handles *nested* model classes such as plane versus cylinder ([Magri & Fusiello](https://re.public.polimi.it/bitstream/11311/1188418/1/paper%20multi%20class%20tlinkage.pdf)), the analogue of a sheet versus its intersection.

In robotics:

- Sturm's articulation learning merges two segments when **BIC(M₁₊₂) < BIC(M₁) + BIC(M₂)** ([Sturm, Stachniss & Burgard, JAIR 2011](https://www.arxiv.org/abs/1405.7705)).
- Subramani and colleagues infer which constraints are present, when, and with what parameters from pose *and* wrench, after removing friction ([arXiv 2010.15916](https://arxiv.org/abs/2010.15916)).
- Debus, Dupont and Howe recovered a 3-D peg-in-hole contact-state sequence, together with the peg's dimensions, **from position sensing only**. They fit all contact-state models in a moving window and fed the residuals to an HMM; the geometry family was known but the parameters were not ([ISER 2002](https://www.bu.edu/biorobotics/publications/02_Iser_pptyestim.pdf)).

Implicit-function learners go further:

- ContactNets learns inter-body signed distances and contact Jacobians with complementarity losses and no mode labels ([Pfrommer et al., CoRL 2020](https://arxiv.org/pdf/2009.11193); [code](https://github.com/DAIRLab/contact-nets)).
- Violation-based losses train systems with "tens of thousands of modes" without known mode boundaries ([Jin et al., L4DC 2022](https://github.com/wanxinjin/Learning-LCS); [Bianchini et al.](https://proceedings.mlr.press/v168/bianchini22a.html)).
- NH-Rep represents a solid as a Boolean tree of neural implicit functions ([Guo et al.](https://arxiv.org/pdf/2209.10191)).
- Point2CAD fits surfaces first and then *intersects* them, so edges and topology emerge from the fits rather than from clustering ([Liu et al., CVPR 2024](https://arxiv.org/pdf/2312.04962)).

**Inference.** The separation function is a minimum over per-feature-pair distances, so fit f(q) = min_k φ_k(q) with K overcomplete:

- **Losses:** a zero-residual loss on contact samples; a positivity loss on separation queries; and a complementarity loss putting τ inside the cone of the active ∇φ_k.
- **Labels:** each sample's mode is its active set {k : φ_k ≈ 0}.
- **Why it handles tangent-continuous boundaries:** two φ_k whose zero sets meet tangentially remain distinct smooth functions.
- **Two-contact strata** are computed as intersections, as in Point2CAD, rather than discovered as thin clusters.
- **Caveat 1:** the decomposition is not unique, so prune with a label-cost or BIC merge.
- **Caveat 2:** building in `peg_hole_analytic.py`'s four-circle family would smuggle in the answer. Keep the φ_k generic, or report the analytic version only as an upper bound.

Sturm's BIC test is also the principled replacement for the repo's patch-merge criterion.

### Signal 4: boundaries along each walk

Detecting a change along a 1-D trajectory is far better posed than finding a boundary in 4-D scatter, and the published tools are cheap:

- **PELT** finds the exact optimum of a penalized segmentation at expected linear cost ([Killick, Fearnhead & Eckley, JASA 2012](https://arxiv.org/abs/1101.1438)), and `ruptures` provides it with kernel and linear-model costs ([Truong, Oudre & Vayatis](https://arxiv.org/abs/1801.00826)).
- **CHAMP** fits an articulation model per segment and was applied to quasi-static object–object contact ([Niekum et al.](https://people.cs.umass.edu/~sniekum/pubs/CPD15.pdf)).
- **Phase HMMs.** Kroemer's phase HMM learned manipulation phases by EM without labels. Four phases gave redundant, flickering states, and the authors recommend merging similar models ([ICRA 2014](https://www.ri.cmu.edu/app/uploads/2019/03/Kroemer_ICRA_2014.pdf)).
- **BP-AR-HMM** infers the number of behaviours and shares them across many series ([Fox et al., Ann. Appl. Stat. 2014](https://arxiv.org/abs/1308.4747)).
- **VHyDRO** (2026) scores contact-mode segmentation with **ARI, change-point F1 and segment purity** ([arXiv 2605.16398](https://arxiv.org/abs/2605.16398)), a set of metrics the repo's benchmark should adopt.

**Inference.** Segment each walk on three signals: ŵ, the turning rate κ = angle(ŵ_t, ŵ_{t+1})/|Δq|, and the twist-span rank. A jump in ŵ marks a crease or a make/break event. A jump in κ marks a C¹ (tangent-continuous) boundary. A rank drop marks a dimension change. The segments then become the nodes of an *unsupervised* stratum graph, the label-free analogue of `stratum_spectral.py`.

## Signed, k-free partitions remove the density and eigengap traps

### Density variation is a sampler artifact

The published estimators split cleanly by how they treat density:

- **Density-insensitive:** TwoNN and Hidalgo ([Allegra et al.](https://publicatt.unicatt.it/retrieve/e309db6f-8864-0599-e053-3705fe0a55db/SCIREP_Hidalgo.pdf)).
- **Robust to varying density:** tensor voting ([Mordohai & Medioni](https://jmlr.org/papers/v11/mordohai10a.html)).
- **Density-exploiting:** Fermat distances deliberately follow "regions with high density" ([fermat on PyPI](https://pypi.org/project/fermat)), and Haro's translated-Poisson mixture clusters by dimension *and* density ([NeurIPS 2006](https://proceedings.neurips.cc/paper/2006/hash/51be2fed6c55f5aa0c16ff14c140b187-Abstract.html)).

**Inference.** Sample density on the peg surface records where the RRT-driven sampler spent its time, not where the modes are. That has three consequences:

- HDBSCAN, ToMATo on density, Fermat distances and the translated-Poisson mixture will cut at sampling-density changes and bridge across well-sampled creases. This is a likely reason HDBSCAN underperformed.
- Methods with a uniform null (HADES, Euclidicity) need the repo's even or balanced subsets first.
- Per-edge *distribution* tests are density-robust by construction. Chen and Friedman's generalized edge-count statistic detects location and scale alternatives on similarity graphs ([Chen & Friedman](https://ar5iv.arxiv.org/html/1307.6294)). Applied to the wrench directions in N(i)∖N(j) versus N(j)∖N(i), it gives each edge a boundary probability that adapts to local noise.

### Balanced cuts are biased against thin modes

**Inference.** Normalized cut, ratio cut, Cheeger cut and total-variation clustering all divide the cut weight by a size or volume term. A thin, low-volume stratum is therefore never a cheap cluster, which explains why large clusters swallow thin modes. Modularity has a known resolution limit as well. In experiments, Louvain left up to 16% of communities disconnected, a defect Leiden repairs; Leiden's Constant Potts Model is resolution-limit-free ([Traag, Waltman & van Eck, Sci. Rep. 2019](https://arxiv.org/pdf/1810.08473)).

### Signed partitioning drops the size term

Minimum-cost multicut "can optimize both the cluster assignment and the number of clusters, and it allows varying cluster sizes, unlike the commonly used spectral clustering approach". It outperformed spectral clustering on motion segmentation ([Keuper, Andres & Brox, ICCV 2015](https://openaccess.thecvf.com/content_iccv_2015/html/Keuper_Motion_Trajectory_Segmentation_ICCV_2015_paper.html)). Lifted multicut adds long-range terms and gave state-of-the-art Princeton mesh decomposition ([Keuper et al.](https://arxiv.org/abs/1505.06973)). Solvers and relatives:

- **RAMA** solves multicut on the GPU with **up to about 1e8 variables in seconds** and gives lower bounds ([Abbas & Swoboda](https://arxiv.org/pdf/2109.01838); [code](https://github.com/pawelswoboda/RAMA), needs CUDA 11.2+).
- **Mutex watershed** needs no seeds or threshold and solves, to global optimality, an objective closely tied to multicut, in near-linearithmic time ([Wolf et al., TPAMI 2021](https://arxiv.org/pdf/1904.12654)).
- **GASP** generalizes signed agglomeration with average, max and min linkage ([Bailoni et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Bailoni_GASP_a_Generalized_Framework_for_Agglomerative_Clustering_of_Signed_Graphs_CVPR_2022_paper.pdf)).
- **python-elf** packages multicut, lifted multicut and mutex watershed via nifty ([elf](https://github.com/constantinpape/elf)).

**Inference.** The number and sizes of the modes then follow from where the edge evidence turns negative, with a single bias β to sweep instead of k. Edge costs come from the per-edge statistics above:

- wrench-direction angle;
- reciprocity residual;
- normal change;
- contact-point jump;
- local-dimension change.

Calibrate them without labels by fitting a two-component mixture (interior versus boundary edges) and setting the cost to the logit of the posterior. Add lifted *repulsive* edges between pose-near, non-adjacent samples whose wrench lines differ. A thin mode then survives whenever its boundary edges are repulsive, because size never enters the objective.

The connectomics pattern of training an edge classifier on other data also applies. Here that means a different peg geometry; it stays unsupervised with respect to the target geometry as long as the target's labels stay out.

### When a count is still required

Several estimators of K come with guarantees or principled rules:

- the LLPD eigengap, under stated conditions ([Little et al.](https://jmlr.org/papers/v21/18-085.html));
- LUND's estimated K ([Murphy & Maggioni](https://arxiv.org/pdf/1810.06702));
- LAPD persistence;
- the Hidalgo posterior;
- label costs;
- BIC merges;
- robust partitions across Markov time from PyGenStability, which also accepts signed graphs ([Arnaudon et al.](https://arxiv.org/pdf/2303.05385)).

**Inference.** The principal-contact lattice adds a structural criterion that the eigengap cannot give: every discovered two-contact cluster must border exactly its constituent single-contact clusters. Huang's enumeration gives the allowed neighbours of each stratum, though no code for it was found.

### If a handful of labels is acceptable

Poisson learning stays stable at very low label rates where Laplace learning degenerates, and `graphlearning` adds active learning ([Calder et al., ICML 2020](https://proceedings.mlr.press/v119/calder20a.html); [GraphLearning](https://github.com/jwcalder/GraphLearning)). **Inference:** one or two labels per mode on the wrench-weighted graph is probably the cheapest route past 0.85.

## Hook modes are kinetic and topological, so connectivity beats clustering

### Hook modes are bottlenecks, not clusters

**Inference.** Free-space hook modes are connected pieces, or metastable sets, of the empty-contact stratum. The wrench is zero there, so contact physics offers nothing, and the useful structure is flow and topology. The repo has already built the two canonical objects:

- spectral clustering on the lattice continuity graph is Meilă–Shi metastability detection;
- ToMATo on separation is the merge tree of clearance.

The open questions are therefore which *function* and which *graph* to use, not which clusterer.

### Kinetic tools

The kinetic toolbox is mature:

- **PCCA+** produces fuzzy metastable memberships from the dominant eigenvectors ([Deuflhard & Weber 2005](https://doi.org/10.1016/j.laa.2004.10.026)).
- **GPCCA** uses a real Schur decomposition, so it works for non-reversible chains ([pyGPCCA](https://pygpcca.readthedocs.io/)).
- **VAMP** remains valid for non-reversible, non-stationary dynamics ([Wu & Noé](https://arxiv.org/abs/1707.04659)), and `deeptime` packages MSMs, VAMP and VAMPnets ([Hoffmann et al.](https://arxiv.org/abs/2110.15013v1)).
- **Transition path theory** committors locate transition regions ([Metzner, Schütte & Vanden-Eijnden](https://doi.org/10.1137/070699500)).
- **Laplacian eigenvectors** of state-transition graphs reflect rooms and the bottlenecks between them ([Mahadevan & Maggioni, JMLR 2007](https://jmlr.org/papers/v8/mahadevan07a.html)). Options built from them discover behaviours that traverse those bottlenecks ([Machado et al., ICML 2017](https://arxiv.org/abs/1703.00956)), and ALLO learns the eigenvectors together with their eigenvalues ([Gomez et al., ICLR 2024](https://arxiv.org/abs/2310.10833v2)).

**Inference.** For the hook, what these add to the existing spectral runs is threefold:

- PCCA+ crispness as a rule for k;
- committor ≈ 0.5 as an explicit "door sample" detector;
- GPCCA if sampler walks, which are non-reversible, replace the lattice graph.

The gain is moderate, because the existing spectral partition already captures the metastable structure.

### Clearance merge trees and doors

GUDHI's ToMATo accepts a manual graph and a manual per-node function ([GUDHI](https://gudhi.inria.fr/python/latest/clustering.html)). Morse–Smale cells carry stability guarantees in Hausdorff distance ([Chen, Genovese & Wasserman, EJS 2017](https://arxiv.org/abs/1506.08826)).

**Inference.** The merge tree of clearance fails in two ways:

- a long chamber with several shallow maxima is over-segmented;
- two physically different modes can meet through a *wide* region with no saddle at all. Threading changes when the tip crosses the loop's spanning disk, and that crossing need not be a clearance bottleneck.

Three fixes follow:

1. **Measure clearance in C-space units.** Run a 6-D distance transform of the free mask under the mm/deg or keypoint-RMS metric, in the spirit of configuration-space distance fields ([Li et al., RSS 2024](https://arxiv.org/abs/2406.01137v1)), instead of workspace millimetres.
2. **Detect doors directly.** Use a vectorized grid bridge test, which marks a free node whose ±r neighbours are both in collision ([OMPL bridge-test sampler](https://ompl.kavrakilab.org/BridgeTestValidStateSampler_8cpp_source.html)). Alternatively, count upper components of each node's 3⁶ neighbourhood as a local saddle test.
3. **Use descending Morse–Smale cells for closest-feature labels.** Their boundaries are discontinuities of ∇separation, so the repo's crease detectors apply unchanged to the [n, r×n] wrench it already computes.

### Convex regions and roadmaps

Convex decomposition gives a third count. VCC seeds IRIS regions from cliques of a sampled visibility graph and adds regions until Monte Carlo coverage reaches α. On a 7-DoF IIWA it cut the count from **482.6 to 46.3 regions**, about 16× faster ([Werner et al., ICRA 2024](https://arxiv.org/html/2310.02875v2)). Its stated failure is that a clique's convex hull can enclose an obstacle. IRIS-ZO certifies regions probabilistically and is about 15.5× faster than IRIS-NP ([Werner, Cohn, Jiang et al.](https://arxiv.org/html/2410.12649v1)); both are in Drake ([Drake IRIS docs](https://drake.mit.edu/doxygen_cxx/group__planning__iris.html)). Slicing over a finite set of orientations gives a conservative 6-D free-space approximation that detects cages and narrow passages ([Varava et al.](https://arxiv.org/abs/2002.02715)).

**Inference.** Greedy max-clique extraction on the existing line-of-sight graph certifies mutual visibility, which a spectral cluster does not. Coverage α is an alternative rule for k. Per-orientation 3-D connected components, linked across neighbouring slices, give a connectivity ground truth.

### Topological threading coordinates

Topology supplies the coordinate that clearance lacks:

- The **Gauss linking integral** has been used as evidence of caging, clasping and hooking for objects with holes ([Stork, Pokorny & Kragic, Humanoids 2013](https://www.csc.kth.se/~fpokorny/static/publications/stork2013c.pdf)).
- Linking-based **topology coordinates** describe how body segments wind around each other ([Ho & Komura](https://diglib.eg.org/items/706d8fb2-da62-4d48-973a-a945a75a0356)).
- **Biot–Savart h-signatures** classify 3-D path homotopy ([Bhattacharya, Likhachev & Kumar, RSS 2011](https://m.roboticsproceedings.org/rss07/p02.html)).
- **Persistent homology** classifies trajectories from free-space samples alone ([Pokorny et al., RSS 2014](https://roboticsproceedings.org/rss10/p54.html)). The relative-homology version is linear in the number of trajectories ([Pokorny, Goldberg & Kragic, ICRA 2016](https://goldberg.berkeley.edu/pubs/icra2016-final-topo-traj-clustering.pdf)).

**Inference.** Compute the signed linking integral of each hook part's centreline with the loop's centreline:

- its integer part tracks the repo's TIP|CROOK|SHAFT threading bits;
- its fractional part passes through ±0.5 at the aperture, which is the door;
- it can serve as a feature, as the ToMATo function, or as a Mapper filter.

For an open curve it is not a homotopy invariant, so treat it as a coordinate, not a certificate.

## Code and scale of the shortlisted tools

| Tool | Failures addressed | Code status | Demonstrated scale | Evidence for this use |
|---|---|---|---|---|
| Hidalgo | Mixed dimension, K | [GitHub](https://github.com/micheleallegra/Hidalgo) MATLAB/Python, no releases; R intRinsic | About quadratic in N; real tests on 31,000 frames | Published dimension separation; **Inference** for the peg |
| GRIDE / TwoNN / local PCA | Mixed dimension | `skdim`, `dadapy` (pip, maintained) | kNN-linear | Published estimators |
| N-D tensor voting | Mixed dimension, creases | No maintained Python found | **Inference:** linear in N at 4×4 tensors | Published per-point dimension |
| Monte-Carlo VCM | Creases, codimension | CGAL is 3-D only; the Monte-Carlo variant is simple to write | **Inference:** O(N_MC log N) | Published for any dimension |
| HADES | Creases, junctions | No public repository; about 50 lines | 6,040 points in 24-D in 5 s | Published transverse theorem |
| GAD / TARDIS | Junctions, boundaries | MATLAB + Ripser / BSD-3 Python ([TARDIS](https://github.com/aidos-lab/tardis)) | Hours on parallel hardware (GAD); a cluster for large examples (TARDIS) | **Inference:** misses two-sheet creases |
| LAPD | Creases, K | [GitHub](https://github.com/HYfromLA/LAPD) | Quasi-linear; tested to 35,000; d ≤ 5 | Published for intersecting flats |
| SSC-OMP / EnSC | Linear subspaces | [Python, sklearn-style](https://github.com/ChongYou/subspace-clustering) | 0.5 million points | **Inference:** poor at codimension 1 |
| RAMA multicut | K, thin modes | pip from GitHub; CUDA ≥ 11.2 | About 1e8 variables in seconds | Published |
| elf/nifty (multicut, lifted multicut, mutex watershed), GASP | K, thin modes | pip/conda `python-elf`; GASP conda-only, small repository | Connectomics scale | Published |
| Leiden-CPM, PyGenStability, Infomap | K, passages | pip packages | Large graphs (not verified at 1e6 here) | Published |
| `graphlearning` (Poisson learning) | Few-label fallback | pip, CI | kNN graphs with annoy | Published |
| `ruptures` PELT | Kinks, walk boundaries | pip | Linear per trajectory | Published |
| `pyparc`, network lasso | Tangent-continuous boundaries | pip / ADMM | Not benchmarked at 1e6 | Published; **Inference** here |
| ContactNets, Learning-LCS | Tangent-continuous boundaries | [contact-nets](https://github.com/DAIRLab/contact-nets), [Learning-LCS](https://github.com/wanxinjin/Learning-LCS) | 60 s of real data; tens of thousands of modes | Published for dynamics; **Inference** here |
| MultiLink, Progressive-X | Model-based modes | [Progressive-X](https://github.com/danini/progressive-x) C++/pybind; MultiLink language unverified | Two-view geometry | Published in vision; **Inference** here |
| deeptime, pyGPCCA | Passages | pip/conda | 1e3–1e5 microstates routine | Published |
| GUDHI ToMATo | Passages, K | Maintained | Up to about 2e9 points | Published |
| Drake IrisZo / clique cover | Passages | Drake (IrisZo experimental; max-clique needs Gurobi or Mosek) | 7-DoF: 46 regions | Published |
| Linking integral | Threading | Trivial to implement | Per pose, O(segments²) | Published as caging evidence; **Inference** as a mode coordinate |

## Ten experiments, ordered by expected gain per day

Each row names the published basis and the inference it tests. Score every experiment with the existing harness: family and mode ARI; `codim_aware_scores.py`; recall on thin modes; and, for walk-based methods, change-point F1 along trajectories. Report jump-type and tangent-continuous boundaries separately.

| # | Experiment (where in the repo) | Failure | Published basis → inference tested | Cost | Success criterion |
|---|---|---|---|---|---|
| 1 | **Break down the supervised ceiling, then add physical features.** In `mode_discovery_bench.py separability`, output mode-level per-class recall and the confusion matrix. Sort confusions into three types: (i) a two-contact mode against its own single-contact constituents; (ii) single-contact modes across a tangent-continuous boundary; (iii) other. Then add feature sets: the wrench line-of-action contact point on the peg surface (undo the 12.5 mm torque scaling) and its hole-frame position; the residual couple; the local τ-cone rank; the ±m-step twist-span rank; and the turning rate κ. | Tangent-continuous, mixed dimension | Intrinsic contact sensing and the contact particle filter → line of action as a feature-identity cue | ½–1 day | If (ii) dominates and the new features push mode balanced accuracy above the 0.761 privileged level, unsupervised exact-mode work has headroom. If (i) dominates, adopt codimension-aware labels as the target. |
| 2 | **Signed, k-free partitioning on the kNN + trajectory graph.** Edge evidence: wrench-direction angle, reciprocity residual \|τ̂_i·Δq̂_ij\|, normal change, contact-point jump from #1, and an edge-count two-sample z-score. Calibrate with a two-component mixture and use logit costs. Run GASP average linkage and mutex watershed (elf), then RAMA multicut on about 1.5e7 edges, sweeping β. | K, thin modes, creases | Multicut, mutex watershed and GASP → repulsive physical edges replace the balanced cut | 1–2 days | Beats NJW at eigengap k on family and mode ARI without choosing k; thin-mode recall rises. |
| 3 | **Linking-integral threading coordinate (hook).** Signed linking integral per hook part (centrelines from `hook_topology.py`) with the loop. Check round(linking integral) against the threading bitmask, then use it as a feature in line-of-sight spectral clustering and as the ToMATo function. | Threading topology | Linking integral as hooking evidence → linking integral as a mode coordinate | Hours | Threading purity and ARI above the current line-of-sight spectral clustering and ToMATo. |
| 4 | **Dimension layer.** Three per-sample codimension estimates: GRIDE or local-PCA eigengap with same-walk neighbours excluded, τ-cone rank, and twist rank. Check their agreement, with `n_active` as a diagnostic. Run Hidalgo (q = 3, ξ = 0.8, K by log-posterior) on 2e4–5e4 evenly thinned points and propagate by kNN vote; first confirm that TwoNN on the analytic two-contact samples gives about 2. | Mixed dimension | Hidalgo, ECoMaNN eigengap → two-contact strata carry their own mass | 1 day | High precision and recall for two-contact samples; exact-mode ARI rises when #2 runs within layers. |
| 5 | **Walk changepoints → segment graph.** PELT (`ruptures`, rbf cost) on [ŵ, κ, twist rank] per trajectory, using `traj`/`step` and parent links. Cut trajectory edges at changepoints, describe each segment, cluster segments with GASP, and coarse-grain the segment transition graph with PCCA+/GPCCA as an unsupervised `stratum_spectral.py`. | Tangent-continuous, K | PELT, CHAMP, VHyDRO metrics → 1-D boundary detection beats 4-D | 1–2 days | Change-point F1 against analytic mode changes along walks; family ARI ≥ 0.85 with no analytic modes as input. |
| 6 | **C-space clearance and doors (hook).** 6-D distance transform of the free mask under the mm/deg metric; ToMATo (manual graph and function) on it compared with workspace separation. Add the grid bridge test and the local saddle test, and compare descending-flow cells with closest-part labels. | Narrow passages | ToMATo, bridge test, Morse–Smale → doors line up with threading transitions | 1 day | Door components coincide with threading transitions; fewer spurious basins than ToMATo on separation. |
| 7 | **Tangent quality and the lever L.** Compare local-PCA, wrench-direction, Monte-Carlo VCM and 4×4 tensor-voting normals against the simulator's contact normals (diagnostics only) as a function of distance to the nearest crease. Tabulate analytic crease dihedral angles over L and pick L. | Creases | Deutsch–Medioni's > π/4 rule; tangent blow-up → wrench normals beat PCA at creases | 1 day | Lower normal error near creases; predicted recoverable creases match where #2 succeeds. |
| 8 | **Kinetic coarse-graining and clique covers (hook).** Lattice-walk microstates (1e3–1e4); PCCA+/GPCCA via deeptime with k from crispness; committors for door samples. Run greedy max-clique on the existing line-of-sight graph with coverage-α stopping and compare with the eigengap k. | Narrow passages, K | PCCA+, transition path theory, VCC → agreement among three independent counts | 1–2 days | Agreement of PCCA+ k, clique count and eigengap k; committor ≈ 0.5 samples sit at the aperture. |
| 9 | **LAPD / Deutsch–Medioni on the 3-D layer.** Remove the crease and junction samples found in #4, run LAPD (d = 3, K by persistence) on ≤ 3.5e4 points, relabel junctions by tangent consistency, and propagate. | Creases, K | LAPD for intersecting flats → also works for folded sheets | 2 days | Recovers crease-type boundaries that #2 misses; K chosen by persistence matches the sheet count. |
| 10 | **Min-of-K implicit constraint model.** ContactNets-style φ_k networks in 4-D (K about 16) with zero, positivity (separation queries) and τ-complementarity losses, label-cost pruning, mode = active set. Keep φ_k generic, and report the analytic four-circle version only as an upper bound. | Tangent-continuous, mixed dimension | ContactNets, NH-Rep, PEARL, Point2CAD → feature-pair functions separate tangent-continuous modes | About 1 week | Separates the type (ii) confusions found in #1; two-contact strata recovered as intersections. |

Experiment #1 comes first because it is cheap and decides the ceiling for everything at mode level. It shows whether exact-mode errors come from label definition or from missing information, and whether the line-of-action feature supplies that information. #2 attacks the family-level gap, which the supervised ceiling shows to be purely algorithmic, and it reuses #1's features as edge evidence. #3 is the cheapest likely win on the hook. #4 and #5 build the dimension and walk-boundary layers that #9 and a refined #2 need. #10 is the most principled answer to tangent-continuous boundaries but the most expensive, so it waits until #1 shows that those confusions dominate. If labels are acceptable at any point, Poisson learning on #2's graph is the fallback.

## Conclusion

The six failures reduce to three kinds of boundary, and each kind needs its own signal. Dimension changes show up as rank: in pose, in the wrench cone, and in the twist span. Creases show up as angle, but only as well as the normals are estimated, and the measured wrench is probably a better normal at a crease than local PCA. Identity changes show up only physically, as where the contact sits on the peg. The repo has spent most of its effort refining the clusterer within the angle signal. Its own separability table says that effort was aimed at the wrong place: the information for families is already in pose, and the failure lies in balanced-cut objectives and density-driven methods that cannot express "a thin mode with sharp boundaries." Moving the evidence into signed edges, and moving boundary detection onto the 1-D walks, changes what the algorithm can represent rather than how hard it searches.

For the hook, mode structure belongs to topology and kinetics rather than to geometry. A linking-integral coordinate turns threading into a scalar that existing Morse tools (ToMATo, merge trees, Mapper) can handle. A C-space clearance field makes doors appear where the hook actually has no room to move. Finally, because no published work scores unsupervised C-space mode discovery against physical labels, the repo's benchmark is itself a contribution. A clean result on even a few of these experiments would be novel.
