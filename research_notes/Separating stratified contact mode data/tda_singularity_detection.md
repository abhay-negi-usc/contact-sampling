# TDA and computational-geometry methods for detecting singular sets, creases, junctions and strata in sampled stratified spaces, and segmenting samples by stratum (applied to unsupervised contact-mode discovery)

Conventions used below. "(BK)" marks a citation from background knowledge that was not re-fetched in this session; the web-search budget ran out before every reference could be re-verified, so treat (BK) items as leads to check. "Peg" means problem (a): 4D reduced coordinates (rho, radial tilt, tangential tilt, depth), 3D single-contact sheets, lower-dimensional multi-contact strata. "Hook" means problem (b): 6-DoF free space near a wire loop, with a separation field on a 1 mm / 1 deg grid.

## 1. Stratification learning via local homology (Bendich et al. 2007; Bendich–Wang–Mukherjee 2012; Skraba–Wang; Nanda 2020): theory and practical implementations

### Takeaway
The local-homology stratification-learning line has solid theory: equivalence of points through local homology transfer maps, sampling bounds, and canonical stratifications via local (co)homology sheaves. But I found no maintained implementation that runs on 1e5+ points. The original algorithm relies on Delaunay/power cells and was only shown on toy grid samples. The Rips-based and sheaf/cosheaf follow-ups exist as theory or pseudocode. A more important limit for the peg: purely topological invariants cannot see a crease where exactly two sheets meet as a wedge, because that set is still a topological manifold. They only fire at sheet boundaries, at junctions of three or more sheets, and at transverse crossings.

### Cited Findings
**Foundations**
- Bendich, Cohen-Steiner, Edelsbrunner, Harer, Morozov (FOCS 2007) use *persistence vineyards*, continuous families of persistence diagrams, to assess the local homology of a point in a stratified space. Other works build related tools: Bendich (2008) and Fasy & Wang (2016) develop persistent local homology variants, Dey, Fan & Wang (2014) use local homology to estimate intrinsic dimension, and Rieck et al. (2020) use predefined stratifications with *persistent intersection homology* (Bendich & Harer 2011). — [von Rohrscheidt & Rieck, ICML 2023, Related Work](https://proceedings.mlr.press/v202/von-rohrscheidt23a.html)
- **Bendich, Wang, Mukherjee, "Local homology transfer and stratification learning" (SODA 2012).**
  - *Core idea.* For points p, q at radius r, the paper compares local structure through the map φ(p,q,r): H(X∩B_r(p), X∩∂B_r(p)) → H(X∩B_r(p)∩B_r(q), boundary), using the persistence diagrams of its kernel and cokernel over a thickening parameter α. Two points are locally equivalent at radius r (x ∼_r y) iff a chain of points joins them in which consecutive maps are isomorphisms; strata are taken as the connected components.
  - *Guarantees.* A "Topological Inference Theorem" infers equivalence from an ε-approximation U of X, with probabilistic bounds on the number of samples needed.
  - *Algorithm.* It uses Delaunay/power-cell decompositions with "lune/moon" subdivisions and nerve complexes, and computes (co)kernel persistence "in time at most cubic in the size of the simplicial complexes."
  - *Experiments.* Only "simple simulations on synthetic data with points sampled from grids" (two intersecting planes).
  - *Authors' own limits.* The algorithm "should be quite slow when the dimensionality of the ambient space is high due to the runtime complexity of Delaunay triangulation." They suggest Rips or witness complexes, PCA, or random projection, and note that "deriving a sufficient sampling condition in practice remains a challenge, especially in high dimensions." They also note that strata of different dimensions need a mixture-model sampling assumption, or else lower strata get no points.
  - [paper PDF](https://sci.utah.edu/~beiwang/publications/Stratification_Learning_BeiWang_2012.pdf); preprint "Towards stratification learning through homology inference" [arXiv:1008.3572](https://arxiv.org/abs/1008.3572)
- **Skraba & Wang, "Approximating local homology from samples".** The local-homology persistence diagrams used for stratification can be approximated with families of Vietoris–Rips complexes, which is presented as making stratification learning practical in high dimensions. — [arXiv:1206.0834](https://arxiv.org/abs/1206.0834)
- **Mileyko, "Another look at recovering local homology from samples of stratified sets" (2018/2019).**
  - Earlier recovery conditions vary from point to point and can degenerate, so the neighbourhood size that preserves local homology need not stay bounded away from zero.
  - The paper gives more general conditions. Tame homology-stratified sets, including Whitney-stratified sets, satisfy them *away from strata boundaries*; "near those boundaries, recovery may not be feasible."
  - The algorithm uses Vietoris–Rips complexes.
  - [arXiv:1811.09033](https://arxiv.org/abs/1811.09033)

**Sheaf, cosheaf and canonical stratifications**
- **Nanda, "Local cohomology and stratification" (FoCM 2020, 20:195–222).** Recovers the *coarsest* stratification of a regular CW complex into cohomology manifolds, by localizing the cell poset around subposets found through cosheaves of local cohomology. The output is a nested sequence of categories, and the process is "amenable to efficient distributed computation." — [arXiv:1707.00354](https://arxiv.org/abs/1707.00354); [ORA record](https://ora.ox.ac.uk/objects/uuid:40cf1db2-2e02-4b18-a3ec-bddf7383aec7)
- **Asai & Shah, "Algorithmic canonical stratifications of simplicial complexes".** Computes the canonical stratification in Nanda's sense using local homology, with a correctness proof and *pseudocode only*. — [arXiv:1808.06568](https://arxiv.org/abs/1808.06568)
- Searches found no software repository for Nanda's algorithm or the Asai–Shah algorithm. — [search summary; arXiv:1707.00354](https://arxiv.org/abs/1707.00354)
- Brown & Wang developed a sheaf-theoretic approach to stratification learning, "from geometric and topological perspectives" (I found the arXiv record but did not read its content). — [arXiv:1712.07734](https://arxiv.org/abs/1712.07734)
- "Stratified Space Learning: Reconstructing Embedded Graphs" (2019) addresses the 1D stratified case and lists future directions. — [arXiv:1909.12474](https://arxiv.org/abs/1909.12474)

**Intersection homology from point clouds**
- Banagl, Mäder, Sadlo (2020) define "stratified formal deformations" and "stratified spines" that preserve intersection homology, implement them, and compute intersection homology of Vietoris–Rips-type complexes of point clouds. A search-engine summary of the abstract adds that computing intersection homology directly from a VR complex usually does not give the correct result near singular points; that statement was not re-checked against the paper. — [arXiv:2005.11985](https://arxiv.org/abs/2005.11985)

**Evidence that practical stratification is fragile**
- A study of local homology on word embeddings reportedly found that full stratification produced discrete clusters, so the authors fell back to a weaker clustering (from the search summary). — [arXiv:1810.10136](https://arxiv.org/abs/1810.10136)

**Exact (non-sampled) stratification of algebraic sets**
- Helmer & Nanda, "Conormal spaces and Whitney stratifications" (FoCM): an implementable algorithm for Whitney stratification of complex projective varieties, via the Lê–Teissier conormal criterion and primary decomposition. It is reported to be orders of magnitude faster than the previous state of the art. — [arXiv:2106.14555](https://arxiv.org/abs/2106.14555)
- An extension handles real algebraic varieties and some semialgebraic sets. — [arXiv:2307.05427](https://arxiv.org/abs/2307.05427)
- Helmer & Mohr give a faster variant and the first algorithm for a *minimal* Whitney stratification. — [arXiv:2406.17122](https://arxiv.org/html/2406.17122v2)

### Inferences
- **Implementations.** None of the 2007–2020 local-homology stratification algorithms has a maintained, scalable implementation that I could find. The usable code in this family computes per-point *annulus* homology instead of transfer maps (GAD, TARDIS; see §2). A practical stand-in for the Bendich–Wang–Mukherjee ∼_r equivalence:
  1. Compute a local-homology signature per landmark, e.g. the number of long H0/H1/H2 bars in an annulus Rips filtration.
  2. Take connected components of the kNN/continuity graph restricted to edges whose endpoints have the same signature.
  3. Propagate labels from the landmarks to all samples.
- **Topology cannot see wedge creases (the most important point for the peg).** The sampled contact set is the boundary of the C-obstacle. Where exactly two single-contact sheets meet at a crease (a convex or concave edge of the C-obstacle), the union is still a topological 3-manifold, so local homology there equals that of a smooth sheet point. This is reasoning, not a cited result. Consequences:
  - Local homology, Nanda's canonical stratification and GAD's bar count would merge the two sheets.
  - They would flag only (i) sheet boundaries, such as sampling-domain limits and the edges of modes that end in free space, (ii) junctions where three or more sheets meet, and (iii) transverse crossings.
  - An acute wedge (e.g., a jamming crease where two opposite-wall sheets close at a small angle) may look like a single sheet *ending in a boundary* once the annulus radius exceeds the wedge gap. GAD would then label it "boundary," not "intersection."
- **Lower strata need their own sampling mass.** Mixture-model sampling is needed for lower strata to be detectable at all. The user's sampler puts mass on thin multi-contact modes ("dense where the sampler found thin contact modes"), so the two-contact strata are likely populated as separate lower-dimensional measures. That makes *local-dimension* signatures (§2: persistent intrinsic dimension, twoNN/lPCA at small k, the growth exponent of kNN radii) cheap and likely more informative than local homology.
- **Units.** All annulus-based local-homology methods depend on the ambient metric. The peg coordinates mix mm and degrees, so the deg-to-mm scale sets the apparent dihedral angles and sheet thickness and needs choosing deliberately (e.g., a lever-arm conversion, as `mode_graph_experiments.py` already does for the hook).
- **Exact stratification as a cross-check (low priority).** The peg/bore contact set is plausibly semialgebraic once rotations are parameterized rationally, so Helmer–Nanda style Whitney stratification could produce an exact stratum list. But the repo already has analytic modes (`peg_hole_analytic.py`), so this only serves as a cross-check, not as an unsupervised method.

### Gaps
- I could not find any code for Bendich–Wang–Mukherjee, Skraba–Wang, Nanda, or Asai–Shah. GitHub was not searched exhaustively because the search budget ran out.
- No benchmark of local-homology stratification on 1e5+ samples, or on stratified spaces with *wedge/boundary* strata as opposed to transverse intersections, was found.
- I did not obtain the content of the Brown–Wang sheaf-theoretic paper (method, code).

## 2. Singularity and anomaly detection in data (GAD, TARDIS/Euclidicity, HADES, Martin–Watson, cyclo-octane) and follow-ups through 2026

### Takeaway
Four methods have usable descriptions:
- **GAD** (Stolz et al., PNAS 2020): annulus H_{k-1} bar counts. MATLAB + Ripser code on GitHub.
- **TARDIS/Euclidicity** (ICML 2023): multi-scale annulus persistence compared against Euclidean model annuli, plus a persistent intrinsic dimension. Python, BSD-3 licence.
- **HADES** (Lim–Oberhauser–Nanda; SIAM journal version, DOI 10.1137/24M1665763): local PCA, then a kernel MMD test against a uniform disk. About O(k²D) per point and seconds for thousands of points. No public code found.
- **Tangent blow-ups** (Petrov–Nabizadeh–Dodik–Solomon, arXiv May 2026): lift each point to (position, tangent projector) so that transversely intersecting sheets separate, with a distance lower bound proportional to the angular gap. Tested up to 200k points.

The peg's sheets are hypersurfaces in 4D, so the blow-up is exactly the "pose + sign-free normal outer product" feature the user already uses. The blow-up paper names the failure the user saw (local PCA averages across sheets near singularities) and adds a second-order (curvature) lift that targets tangent-continuous boundaries. HADES is the only topology-free detector that clearly scales to 1e5–1e6 points. However, its uniform-disk null will misfire on the user's very non-uniform densities unless they are equalized first.

### Cited Findings
**Geometric Anomaly Detection (GAD): Stolz, Tanner, Harrington, Nanda, PNAS 117(33):19664–19669, 2020**
- *Inputs.* Point set P ⊂ R^n, the manifold dimension k, and annulus radii. For each point y, the annulus is A_y = {x : r ≤ ‖x−y‖ ≤ s}. — [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC7443892)
- *Classification.* Compute the (k−1)-dimensional barcode of A_y (Rips) and count the bars longer than (s−r): 0 bars means boundary, 1 means k-manifold point, 2 or more means intersection/singular point. Lower strata come from re-running on P − P_man with k → k−1. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7443892)
- *Caveats stated by the authors.* The output may not be a genuine stratification because "the algorithm does not check for normal triviality." Some non-generic intersections are misclassified. There is no formal correctness theorem; robustness is inherited from persistence stability. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7443892)
- *Parameter oddity.* The paper reports (r,s) = (0.4, 0.25) for cyclo-octane and (2, 1.5) for the Henneberg surface, which have r > s and conflict with the stated 0 < r < s. The ordering is probably swapped in the text. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7443892)
- *Data and code.* Cyclo-octane: 6,040 points in R^24, subsampled from more than 1M conformations. Henneberg surface: 5,456 points. The implementation is MATLAB (rangesearch) + Ripser, and the per-point computations "can—and should—be computed in parallel." Code and data: https://github.com/stolzbernadette/Geometric-Anomalies. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7443892)
- *Runtime.* The HADES authors report, from a private communication, that GAD "took at least several hours on parallel processing." — [HADES, arXiv:2311.04171](https://arxiv.org/html/2311.04171)

**TARDIS / Euclidicity: von Rohrscheidt & Rieck, ICML 2023 (PMLR 202)**
- *Persistent local homology.* Uses the intrinsic annulus B_r^s(x) and a tri-filtration (r, s, Rips scale t). The persistent intrinsic dimension (PID) is i_x(ε) = max{i : ∃ r<s<ε with PH_{i−1}(V(B_r^s(x),•)) ≠ 0}. — [paper](https://proceedings.mlr.press/v202/von-rohrscheidt23a.html)
- *Euclidicity.* The expected distance between a point's persistent local homology and that of Euclidean model annuli of the same size. It is implemented as the mean bottleneck distance of fibred Rips barcodes over a grid of (r,s). — [paper](https://proceedings.mlr.press/v202/von-rohrscheidt23a.html)
- *Guarantees.*
  - Stability: ½·d_B ≤ max{ε1, ε2} under changes ε1, ε2 in r, s.
  - PID recovers the intrinsic dimension on smooth manifolds, and per stratum on stratified simplicial complexes (Thm 2, Cor 1).
  - The Euclidicity approximation error is bounded for smooth manifolds (Thm 3).
  - [paper](https://proceedings.mlr.press/v202/von-rohrscheidt23a.html)
- *Parameters.* s_max is the distance to the k-th neighbour (k = 50 suffices in their experiments; higher intrinsic dimension may need more). r_min is the smallest nonzero neighbour distance. s_min and r_max are set at the ⌊k/3⌋-th neighbour, with 20 steps. — [paper](https://proceedings.mlr.press/v202/von-rohrscheidt23a.html)
- *Limitations stated by the authors.* The Rips complex grows exponentially with dimension, memory may block high-dimensional data, and reliability depends on a correct intrinsic-dimension estimate. — [paper](https://proceedings.mlr.press/v202/von-rohrscheidt23a.html)
- *Position relative to GAD.* The authors call GAD the closest method, but note that GAD needs the intrinsic dimension and uses one global scale. — [paper](https://proceedings.mlr.press/v202/von-rohrscheidt23a.html)
- *Code.*
  - https://github.com/aidos-lab/tardis: BSD-3-Clause, 174 commits, 45 stars, tested on Python 3.8/3.9. It has a CLI (`cli.py` with annulus flags, or automatic `-k`, `-q` query count, `-d` dimension, `--num-steps`) and a "preliminary API" (`tardis/api.py`).
  - The README warns the wedged-spheres example "might require a long runtime on an ordinary machine" and that the authors used a cluster. — [GitHub](https://github.com/aidos-lab/tardis)
  - The docs call the project "a constant work in progress." — [docs](https://tardis-tda.readthedocs.io/en/latest/)

**HADES: Lim, Oberhauser, Nanda (arXiv Nov 2023; SIAM, DOI 10.1137/24M1665763)**
- *Score.* The local neighbourhood of radius r is projected by PCA to an estimated dimension d̂ (the number of components explaining a fraction η of the variance). The score is the kernel MMD between the projected local measure and the uniform measure on the unit d̂-disk. p-values come from a Monte-Carlo null. — [arXiv HTML](https://arxiv.org/html/2311.04171)
- *Parameters.* r, η ∈ [0.7, 0.9], kernel parameter α ∈ [0.3, 0.7], and r ∈ [1.5r̃, 5r̃] from a local-scale detection step. — [arXiv HTML](https://arxiv.org/html/2311.04171)
- *Complexity.* O(k²D) for PCA plus O(k²d̂) for MMD per neighbourhood, i.e. linear in the ambient dimension D, versus an estimated O(k^(3d+3)) for persistent homology. — [arXiv HTML](https://arxiv.org/html/2311.04171)
- *Guarantee (Thm 7).* For two smooth compact d-manifolds meeting transversely (tangent spaces sharing a d0-dimensional subspace, principal angles ≥ φ), with high probability for large n: points within c_A·r of the intersection score above 2ξ and points farther than c_B·r score below ξ. The guarantee covers only this two-manifold transverse case. — [arXiv HTML](https://arxiv.org/html/2311.04171)
- *Timings (2018 laptop).*
  - Cyclo-octane, 6,040 points in R^24: "5 seconds."
  - Road images with 45k–200k nonzero pixels: 6–31 s, using 10% of pixels.
  - MNIST classes in 100D: 30–45 s.
  - Synthetic sets of 5k–10k points: 48–195 s.
  - The method is slower than Ripser in low dimension, but "the situation is quickly reversed in high-dimensional data."
  - [arXiv HTML](https://arxiv.org/html/2311.04171)
- *Results.* ROC AUC ≥ 0.89 on high-dimensional synthetic families. The SUPC manifold-hypothesis test reaches AUC 1.00 at n = 8,000. It finds the cyclo-octane intersection rings directly in 24D. — [arXiv HTML](https://arxiv.org/html/2311.04171)
- *Limitations stated by the authors.* The test only signals "what is not like a disk," hyperparameters come from a heuristic, and PCA dimension estimation could be improved. — [arXiv HTML](https://arxiv.org/html/2311.04171)
- *Code.* Implemented in Python; no repository link was found on arXiv or by search. — [arXiv abs](https://arxiv.org/abs/2311.04171)

**Tangent blow-ups: Petrov, Nabizadeh, Dodik, Solomon, "Tangent Blow-Ups for Processing Non-Manifold Geometry" (arXiv:2605.18215, May/June 2026)**
- *Lift.* Each sample is lifted to (ambient position, tangent plane in the Grassmannian), with metric d² = ‖p−q‖² + (α/2)‖P_U − P_V‖²_F; mostly α = 1. Iterating the lift separates second-order (curvature) differences. — [arXiv HTML](https://arxiv.org/html/2605.18215)
- *Theorem.* For d-submanifolds meeting at x0 with tangent-plane chordal distance δ, the lifted distance near x0 is at least (1/√2)·√α·δ. The construction is stated for Gr(d,n) in general dimension. — [arXiv HTML](https://arxiv.org/html/2605.18215)
- *Segmentation and scale.* The 10 smallest eigenvectors of the normalized lifted Laplacian are row-normalized and clustered with DBSCAN (ε = 0.2). Results are qualitative only, with no metrics. Up to 200k points; level-1 and level-2 lifts took 38 s and 77 s on one model. — [arXiv HTML](https://arxiv.org/html/2605.18215)
- *Limitations stated by the authors.*
  - Tangent estimates near singularities: PCA "averages across intersecting components." Positional noise is tolerated better than normal noise.
  - One tangent per point is assumed, and tangential contacts leave cross-sheet contamination.
  - The lifted dimension grows quickly (kD-tree bottleneck).
  - No code link in the text I read.
  - [arXiv HTML](https://arxiv.org/html/2605.18215)

**Martin–Watson and the cyclo-octane conformation space (the closest analogue to a sampled configuration space)**
- Martin & Watson, "Non-manifold surface reconstruction from high dimensional point cloud data," *Computational Geometry* 44(8):427–441 (2011); earlier Sandia report SAND2009-3420J. — [OSTI record](https://www.osti.gov/biblio/958390); [SUNY record](https://researchconnect.suny.edu/en/publications/reconstructing-manifold-and-non-manifold-surfaces-from-point-clou/)
- Freedman's manifold reconstruction method failed on cyclo-octane because of self-intersections. The ring-atom dataset has 1,031,644 points in R^24, and the authors produced triangulations with 6,040–13,194 vertices. The companion paper is "Topology of cyclo-octane energy landscape," *J. Chem. Phys.* 132 (2010), and the reconstruction software is NMTRI. — [ICERM PH tutorial slides](https://app.icerm.brown.edu/materials/Abstracts/tri18-2-tml/Tutorial-_Persistent_homology.pdf)
- The cyclo-octane space contains a sphere and a Klein bottle meeting in two rings, which is what GAD and HADES detect. — [HADES](https://arxiv.org/html/2311.04171); [GAD PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7443892)
- Bauer & Lenzen (2025) infer a cell structure on the cyclo-octane conformation space by partitioning it into subsets that share symmetry patterns, using dimensionality reduction and TDA. They find the *unlabeled* conformation space contractible, unlike the labeled one. No code link was found. — [arXiv:2502.20149](https://arxiv.org/abs/2502.20149)

**Other 2023–2026 follow-ups and adjacent methods**
- Andersson & Avelin derive explicit bounds on the graph Laplacian applied to functions near singularities, build tests for singularities and estimate their geometric properties. Published in J. Comput. Math. Data Sci. 14 (2025), article 100113; v4 on arXiv in Feb 2026. — [arXiv:2301.00201](https://arxiv.org/abs/2301.00201)
- "Topologically Relevant Landmark Selection on Point Clouds" (arXiv:2610.00019, about Oct 2026) selects landmarks by their contribution to global homology. Its introduction surveys local-PH singularity detection; it is a subsampling method. — [arXiv:2610.00019](https://arxiv.org/html/2610.00019v1)
- Brown et al. (2023) proposed the "union of manifolds hypothesis," which allows the dimension to vary across components but excludes singularities. — [TARDIS related work](https://proceedings.mlr.press/v202/von-rohrscheidt23a.html)
- Arias-Castro, Lerman, Zhang, "Spectral clustering based on local PCA" (JMLR 18, 2017): a kNN graph weighted by the discrepancy between local principal subspaces, then spectral clustering. Unlike distance-only spectral methods it "is able to resolve intersections." Guarantees are proven for simplified variants. — [JMLR](https://jmlr.org/papers/v18/14-318.html); [arXiv:1301.2007](https://arxiv.org/abs/1301.2007)
- The TARDIS dimension comparison used lpca, twoNN and DANCo from the `scikit-dimension` toolkit. On S¹∨S², PID gave 1D points 1.12±0.24 and 2D points 1.95±0.06, versus twoNN's 1.00±0.07 and 1.90±0.36. — [TARDIS paper, Table 1](https://proceedings.mlr.press/v202/von-rohrscheidt23a.html)

### Inferences
- **Peg: which detector sees which stratum** (reasoning from the definitions above).
  - *GAD with k = 3 (H2 of 4D annuli).* One long H2 bar on smooth sheets *and* on two-sheet wedge creases, so creases are missed. Zero bars where a sheet ends, and possibly at acute jamming wedges at coarse scale. Two or more bars only where three or more sheets meet or sheets cross.
  - *Euclidicity.* Fires weakly on sharp wedges (folding the annulus shortens Rips persistence) and strongly on boundaries and junctions.
  - *HADES.* The PCA-projected measure of a folded neighbourhood is compressed along the fold, so HADES probably fires on sharp creases even though its theorem covers only transverse intersections. Not verified.
  - *Tangent blow-up.* Separates any crease with a normal jump δ, with margin ∝ √α·δ. It is blind to tangent-continuous boundaries at level 1; level 2 (curvature) is needed there.
- **The peg normal feature already is a level-1 tangent blow-up.** For hypersurfaces in R^4, ‖P_U − P_V‖_F² = ‖nnᵀ − mmᵀ‖_F², so `mode_discovery_bench.py`'s "normal (sign-free outer-product upper triangle)" *is* the level-1 lift. The Petrov et al. theorem is a formal justification for it, and their stated failure mode (PCA averaging across sheets) explains residual mixing near creases.
- **Better normals for that lift.** Use normals that do not average across sheets: robust/Hough/VCM normals (§3), or the *measured wrench direction*. For a frictionless single contact, the contact force direction is the C-space normal of the contact sheet; with friction it lies within the friction cone around it.
- **Add a level-2 (curvature) lift** for the tangent-continuous boundaries, where the normal turns smoothly but curvature likely jumps (for example, an edge sliding around the rim versus a face on the mouth).
- **Density.** HADES and Euclidicity both compare against *uniform* model measures (uniform disk, uniform Euclidean annulus). With the sampler's very non-uniform density, density gradients inside a ball will be scored as "non-Euclidean." Equalize density first (the bench's `balanced` subset, farthest-point/voxel thinning, or inverse-density weights) before trusting either score. This is an inference: neither paper tests strongly non-uniform densities.
- **Scalability at 1e5–1e6 points in 4–6D.**
  - *HADES and tangent blow-up:* yes. kNN + PCA + MMD per point is O(k²) work, and 4–6D kNN with cKDTree is cheap.
  - *GAD and Euclidicity:* not for every point. H2 of Rips annuli for every sample, and for Euclidicity a 20×20 (r,s) grid plus Euclidean model resamples, is far heavier; the authors already needed hours to a cluster on thousands of points. Score about 1e4 farthest-point landmarks and propagate labels over the kNN graph instead.
- **Positive-mass lower strata (cheapest first experiment).** Because the user's lower strata carry sampling mass, a *multi-scale local dimension profile* per point should separate 3D sheets from 2D/1D multi-contact strata. Candidates: PID from TARDIS, twoNN/lPCA at several k, or the slope of log k against log r_k. This costs one kNN query and is robust to density level, though not to anisotropy.
- **Hook.** Free space is full-dimensional (6D), so singularity detectors on free-space *samples* fire at its boundary (contact) and at *narrow passages*, where the free region becomes thinner than the sampling scale and looks locally lower-dimensional (a "neck"). On the grid, the clearance field already measures narrowness directly (§4), so these detectors add little for the hook except a sample-based cross-check.
- **Symmetry as an a-priori stratum source.** The Bauer–Lenzen cyclo-octane result (cells defined by symmetry patterns) suggests listing the fixed sets of any residual symmetry in the reduced coordinates (e.g. rho = 0, or tangential tilt = 0 if the sign of tangential tilt is a symmetry) as candidate strata or boundaries before the data-driven search.

### Gaps
- No public HADES code was found, so it would have to be reimplemented (about 50 lines: kNN, PCA, MMD against Monte-Carlo uniform-disk samples). I found no tangent blow-up code either.
- No method in this family has been evaluated on *wedge* creases or *tangent-continuous* stratum boundaries; every guarantee is for transverse intersections, boundaries or cone points.
- Specific 2024–2026 citing works of Euclidicity/GAD beyond those listed could not be enumerated (search budget exhausted).
- I did not verify the exact *J. Chem. Phys.* article number for Martin et al. 2010, or whether NMTRI is still downloadable.

## 3. Sharp-feature / crease detection and robust normals: which methods extend beyond 3D?

### Takeaway
Two families have dimension-general theory and suit the 4D peg directly:
- **Voronoi Covariance Measure (VCM).** Mérigot et al. give a Monte-Carlo version "applicable in any dimension," and the δ-VCM of Cuel et al. adds robustness to outliers via a distance-to-measure. Its eigenstructure estimates the *normal cone*, whose rank equals the number of independent active contact normals: 1 on a sheet, 2 on a two-contact crease, 3 at a three-contact corner. That is a direct geometric proxy for contact-mode codimension.
- **Gauss-map clustering of neighbour normals** (Weber–Hahmann–Hagen) is conceptually dimension-free.

Hough normal voting (Boulch–Marlet), jets (Cazals–Pouget/CGAL), difference-of-normals, and robust MLS/RIMLS are implemented only for 3D, although their ideas generalize. Learned edge detectors (EC-Net, PIE-Net, NerVE, DEF) are 3D-only and trained on CAD shapes (ABC), so they are not applicable.

### Cited Findings
**Voronoi Covariance Measure**
- Mérigot, Ovsjanikov, Guibas, "Voronoi-based curvature and feature estimation from point clouds" (IEEE TVCG 17(6), 2011). Error is bounded by the Hausdorff distance between the point cloud and the surface. The method works in 2D/3D and is parallelizable, and "a Monte-Carlo version of our method … is applicable in any dimension." — [paper PDF](https://www.lix.polytechnique.fr/~maks/papers/voronoi.pdf); [IP Paris record](https://researchportal.ip-paris.fr/en/publications/voronoi-based-curvature-and-feature-estimation-from-point-clouds/)
- Cuel, Lachaud, Mérigot, Thibert, "Robust geometry estimation using the generalized Voronoi covariance measure" (SIAM J. Imaging Sci. 8(2):1293–1314, 2015). Extends VCM to any distance-like function δ (e.g., distance-to-measure). The δ-VCM resists both Hausdorff noise and outliers, with experiments on normals, curvature and sharp-feature detection. — [HAL](https://hal.archives-ouvertes.fr/hal-01058145)
- VCM thresholding and Edge-Aware Resampling are in CGAL's point-set processing and served as classical baselines for PIE-Net. — [PIE-Net, arXiv:2007.04883](https://arxiv.org/abs/2007.04883)

**Gauss-map clustering (Weber, Hahmann, Hagen)**
- Clusters the Gauss map of each local neighbourhood to discard points unlikely to lie on a sharp feature, with a single sensitivity parameter (later made adaptive), working directly on the point cloud without reconstruction. — [Dagstuhl OASIcs VLUDS 2010](https://drops.dagstuhl.de/entities/document/10.4230/OASIcs.VLUDS.2010.90); [HAL](https://hal.archives-ouvertes.fr/hal-00568051)
- Weber's 2011 PhD thesis covers the method in more detail. — [thesis](https://kluedo.ub.rptu.de/files/2809/c_weber_phd.pdf)

**Hough normals (Boulch & Marlet, SGP 2012)**
- *Core idea.* A randomized Hough transform: random point triples vote for plane normals in an accumulator treated as a discrete probability distribution, and the normal is the accumulator's maximum. Statistical bounds fix the number of triples (confidence α, distance δ), with a confidence-interval early stop and randomized accumulator rotations against discretization. — [paper PDF](https://imagine.enpc.fr/~marletr/publi/SGP-2012-Boulch-Marlet.pdf)
- *Cost.* O(n log n) overall (kd-tree plus per-point work); times are comparable to jet fitting and to the Voronoi method NormFet. — [paper PDF](https://imagine.enpc.fr/~marletr/publi/SGP-2012-Boulch-Marlet.pdf)
- *Code.* Header-only C++ (Eigen + nanoflann). Defaults: 200 neighbours, 700 planes, 5 rotations. — [aboulch/normals_Hough](https://github.com/aboulch/normals_Hough) (found via a redirect link; not opened directly)
- A learned successor exists (CGF 2016). — [Eurographics DL](https://diglib.eg.org/handle/10.1111/cgf12983)
- *Survey of other estimators (from the same paper's related work).*
  - Regression methods (PCA planes, spheres, jets) "tend to smooth sharp features."
  - MLS and robust local kernel regression (Öztireli et al. 2009, i.e. RIMLS) "can retrieve sharp features, but they depend on a reliable prior estimation of input normals."
  - Dey's Voronoi pole method preserves features but "is quite sensitive to noise."
  - Li et al.'s robust-noise-scale RANSAC handles noise and outliers but is "not very fast (typically around half an hour for 1.5 million points)."
  - Most methods do not handle sampling anisotropy at edges.
  - [paper PDF](https://imagine.enpc.fr/~marletr/publi/SGP-2012-Boulch-Marlet.pdf)

**Jets, difference-of-normals, robust MLS**
- *Jets (Cazals & Pouget, SGP 2003).* Polynomial fitting of osculating jets. CGAL's `Monge_via_jet_fitting` takes `Point_3` input (3D surfaces only), and a d-jet needs at least (d+1)(d+2)/2 points. — [CGAL manual](https://doc.cgal.org/4.8/Jet_fitting_3/classCGAL_1_1Monge__via__jet__fitting.html)
- *Difference of Normals (Ioannou et al., 3DIMPVT 2012).* A multi-scale operator: the difference between normals estimated at two radii. Built for large *3D* unorganized clouds (urban LIDAR). — [arXiv:1209.1759](https://arxiv.org/abs/1209.1759)
- *Fleishman, Cohen-Or, Silva (2005), robust MLS with sharp features.* Robust forward-search fitting that splits a neighbourhood into piecewise-smooth regions; the user already uses a "neighbourhood shifting" variant of it. — (BK) [ACM TOG 24(3), DOI 10.1145/1073204.1073227](https://doi.org/10.1145/1073204.1073227)
- *Öztireli, Guennebaud, Gross (2009), RIMLS.* "Feature preserving point set surfaces based on non-linear kernel regression." — (BK) [CGF 28(2), DOI 10.1111/j.1467-8659.2009.01388.x](https://doi.org/10.1111/j.1467-8659.2009.01388.x)

**Learned edge detectors (all 3D)**
- *EC-Net (ECCV 2018).* A patch network jointly predicts 3D coordinates and each point's distance to edges; trained on virtual scans of meshes. — [arXiv:1807.06010](https://arxiv.org/abs/1807.06010)
- *PIE-Net (NeurIPS 2020).* Proposes edge and corner points, then fits parametric curves (lines, circles, B-splines). Trained on the ABC CAD dataset; outperforms VCM, EAR and EC-Net (self-reported). — [arXiv:2007.04883](https://arxiv.org/abs/2007.04883)
- *NerVE (CVPR 2023).* Volumetric neural edge grid, then piecewise-linear curves, then a graph search; evaluated on ABC. — [arXiv:2303.16465](https://arxiv.org/abs/2303.16465)
- *DEF.* "Deep Estimation of Sharp Geometric Features in 3D Shapes." — [arXiv:2011.15081](https://arxiv.org/pdf/2011.15081)

**Boundary detection (dimension-general statistical theory)**
- Calder, Park, Slepčev, "Boundary estimation from point clouds: algorithms, guarantees and applications" (2021/22). Estimators of the boundary normal, the distance to the boundary, and a boundary-strip membership test, with rigorous error estimates; also used for PDEs on point clouds. — [arXiv:2111.03217](https://arxiv.org/abs/2111.03217)

**Second-order lift**
- The tangent blow-up's second lift separates "second-order geometry (curvature)," e.g. a line versus a tangent parabola. — [arXiv:2605.18215](https://arxiv.org/html/2605.18215)

### Inferences
- **VCM normal-cone rank as a direct contact-mode codimension detector (top recommendation from this section).**
  - *Why it fits.* On the boundary of the C-obstacle, the normal cone at a k-contact configuration is spanned by the k contact-normal directions (frictionless), and its dimension is the number of independent contacts. The VCM integrates the covariance of the Voronoi cells of samples over an offset, so the eigenvalue spectrum of the per-point VCM tensor gives normal-cone dimension 1, 2 or 3.
  - *How to compute in 4D.* Use the Monte-Carlo VCM: sample offset points within distance R of the cloud and assign each to its nearest sample, which avoids building a 4D Voronoi diagram. The δ-VCM with distance-to-measure adds robustness to the density spikes.
  - *Cost.* O(N_MC log N) with a kd-tree, feasible for 1e6 points in 4D.
  - *Limit.* Tangent-continuous boundaries leave the normal-cone rank unchanged, so VCM will not separate them.
- **Physical analogue of VCM.** The local covariance of *measured wrench directions* (in the reduced, lever-scaled frame) over a neighbourhood also estimates the normal cone: rank about 1 on a sheet and about 2 on a crease. Friction inflates it by the friction-cone half-angle. A rank threshold that scales with μ, or the use of the friction-free "contact" wrench where it is available for diagnosis, could give a crease detector that does not rely on PCA geometry at all.
- **Gauss-map clustering generalizes cleanly.** Per point, cluster neighbours' (robust) normals on S³ or RP³ (sign-free). One cluster means a sheet, two a crease, three or more a corner. This is the discrete counterpart of VCM, and it can use the user's existing local-PCA normals. Near creases, compute those normals from *shifted* neighbourhoods, as the user already does, to avoid the averaging failure.
- **Hough voting in 4D.** The accumulator is S³ (or RP³), and random 4-tuples define hyperplanes. Bin count grows roughly as (1/Δθ)³, so it is still feasible at about 5–10° resolution (inference). The *multimodality* of the accumulator, not only its argmax, is the crease signal.
- **Other methods.**
  - *Jets / quadrics in 4D* (graph of a function of 3 variables) need at least 10 points for degree 2 and 20 for degree 3. Curvature-jump detection (the level-2 lift) is the only geometric signal for tangent-continuous boundaries, but it suffers the same neighbourhood mixing.
  - *Difference-of-normals in 4D* is trivial to implement: the angle between normals at two scales. It is a cheap first-pass crease score.
  - *Learned 3D edge detectors are not applicable* to 4D C-space, both because of the dimension and because they were trained on CAD (ABC).
- **Calder et al.'s boundary-strip test** is the principled tool for the "sheet ends here" stratum: modes that end in free space, sampling-domain edges, and the acute-wedge regime from §1.

### Gaps
- I did not verify whether the δ-VCM sharp-feature results (as opposed to normals) are stated for arbitrary dimension, nor whether any public non-CGAL VCM code exists for d > 3.
- No published 4D or higher crease detector on C-space samples was found.
- Fleishman 2005 and RIMLS details were not re-fetched (BK).
- Calder et al.'s code availability is unknown.

## 4. Morse-theoretic and mode-seeking segmentation (ToMATo, Morse–Smale clustering, Mapper/Reeb/Ball Mapper, discrete Morse on the separation field)

### Takeaway
For the hook, the right objects are the **merge tree (0-dimensional superlevel persistence) of clearance**, whose saddles are the narrow passages, and the **Morse–Smale cells of the separation field**:
- *Ascending cells* (flow up to clearance maxima) give free-space "chambers."
- *Descending cells* (flow down to contact) give "which features are closest," because following −∇separation from a free pose ends on the closest feature.

GUDHI's ToMATo accepts a manual graph and manual function, so it computes the first object directly at billions-of-points scale. kNN-graph Morse–Smale approximations exist (R `msr`; Morse crystals) with statistical stability theory (Chen–Genovese–Wasserman). Mapper/Ball Mapper with a threading filter gives a mode graph. For the peg, density-based mode-seeking is the wrong signal (density reflects the sampler, not the modes). Run ToMATo or Morse–Smale on a *stratum indicator* such as normal-cone rank or Euclidicity instead.

### Cited Findings
**ToMATo (GUDHI)**
- *Interface.* `gudhi.clustering.Tomato` takes `graph_type` 'knn', 'radius' or 'manual' (your own neighbour lists) and `density_type` 'logDTM', 'DTM', 'KDE', 'logKDE' or 'manual' (your own per-node function via `weights`). Merging is set by `n_clusters` or `merge_threshold` (prominence).
- *Outputs.* `labels_`, `leaf_labels_` (before merging), and `diagram_` (the merge-tree persistence diagram).
- *Scale.* Points are limited to about 2 billion, and `n_jobs` parallelizes. DTM can overflow in high dimension, and KDE is slow for many points.
- [GUDHI docs](https://gudhi.inria.fr/python/latest/clustering.html)

**Morse–Smale complex, statistical view**
- Chen, Genovese, Wasserman, "Statistical inference using the Morse-Smale complex" (EJS 11(1):1390–1433, 2017). The basins of attraction of a density's modes are the population clusters, and each cluster is a union of Morse–Smale cells. The paper proves stability of the complex in Hausdorff distance (Thm 1) and a risk bound for mode (mean-shift) clustering (Thm 2), and covers Morse–Smale regression and two-sample tests. — [arXiv:1506.08826](https://arxiv.org/abs/1506.08826); [Project Euclid](https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-11/issue-1/Statistical-inference-using-the-Morse-Smale-complex/10.1214/17-EJS1271.short)

**Morse–Smale complex on a kNN graph**
- Gerber & Potter's `msr` R package (JSS 50(2), 2012) is a "discrete Morse-Smale complex approximation based on kNN graph," used for partition-based regression and visualization in arbitrary dimension. — [JSS](https://www.jstatsoft.org/article/view/v050i02); [rdrr](https://rdrr.io/cran/msr)
- Harvey, Rübel, Pascucci, Bremer, Wang cluster points into "Morse crystals" by a discrete gradient. These crystals may contain loops, which they fix with a Reeb-graph subdivision. — [PDF](https://birg.cs.wright.edu/papers/ms_reebgraph.pdf)

**Mapper and Ball Mapper**
- Ball Mapper (Dłotko 2019) is a Mapper-inspired shape summary built from an ε-net cover of balls; it needs only a radius ε instead of filter functions and cover choices. — [arXiv:1901.07410](https://arxiv.org/abs/1901.07410)
- Implementations: Python [pyBallMapper on PyPI](https://pypi.org/project/pyBallMapper/0.3.1) (repo github.com/dioscuri-tda/pyBallMapper, per [Rudkin tutorial arXiv:2505.03022](https://arxiv.org/pdf/2505.03022v2.pdf)) and R [BallMapper on CRAN](https://cloud.r-project.org/web/packages/BallMapper/index.html).
- Mapper itself: Singh, Mémoli, Carlsson, Eurographics SPBG 2007. — (BK) [DOI 10.2312/SPBG/SPBG07/091-100](https://doi.org/10.2312/SPBG/SPBG07/091-100)

**Spectral/Hodge clustering by global topological role**
- *TPCC (Grande & Schaub, ICML 2023, PMLR 202:11683–11697).* Clusters points by their contribution to global topological features, via spectral properties of a simplicial complex built on the cloud. Code: RWTH GitLab (netsci group). — [PMLR](https://proceedings.mlr.press/v202/grande23a.html)
- *TOPF (2024/ICML 2025).* A successor with point-level topological features; `pip install topf` (macOS/Linux), [GitHub](https://github.com/vincent-grande/topf). The TOPF authors say TPCC "is unstable against some form of noise … and is computationally far more expensive." — [arXiv:2406.02300](https://arxiv.org/abs/2406.02300)

**Persistent cohomology on free-space samples**
- Pokorny & Kragic, "Data-driven topological motion planning with persistent cohomology" (RSS 2015). Needs only samples of free configuration space; works on arbitrary filtrations, so planning can be restricted to a sub- or superlevel set of a function such as cost or clearance; "scales well in low dimensions." — [RSS XI p49](https://roboticsproceedings.org/rss11/p49.html); [PDF](https://www.csc.kth.se/~fpokorny/static/publications/pokorny2015a.pdf)

### Inferences
- **Hook passages from the merge tree.**
  - *What to compute.* The 0-dimensional superlevel persistence of clearance on the grid continuity graph is the same thing as a maximum spanning tree with Kruskal merges. Each merge edge is a *passage*, and its clearance is the passage width. ToMATo with `graph_type='manual'`, `density_type='manual'` and `weights=separation` computes exactly this (the user has run ToMATo on continuity edges).
  - *Why it can disagree with the physical labels.* A long, flat chamber can contain several shallow clearance maxima (over-segmentation, fixed by `merge_threshold`), and two physical modes can join through a *wide* region with no saddle at all (under-segmentation). For example, the "threading" state changes as the hook tip crosses the loop's spanning disk, which need not be a clearance bottleneck.
- **Hook closest-feature modes from the descending flow.**
  - The "which hook part and which eyelet part are closest" labels are the cells of a generalized Voronoi diagram in C-space. Their boundaries are where the separation field's gradient jumps (bisectors), not saddles of its value.
  - The *descending* Morse–Smale cells (downhill flow to contact, which is what `prepare-contact` does by sliding along the normal) give this partition directly.
  - Detect the boundaries as gradient-direction discontinuities, the same machinery as §3 applied to ∇separation (= the [n, r×n] wrench the repo computes): local covariance of gradient directions has rank 2 or more at bisectors.
- **Local saddle test on the grid (cheap discrete Morse).** For node x, count the components of {sep > sep(x)} within its 3^6 grid neighbourhood. Two or more upper components mark a 1-saddle, a candidate passage. This is O(3^6) per node and needs no global complex. A full cubical Morse–Smale complex in 6D is likely memory-prohibitive on a fine grid (inference). (BK) GUDHI's `CubicalComplex` accepts arbitrary dimension, but the number of cells grows about 3^d per vertex.
- **Hook mode graph from Mapper/Reeb.** Use a physically meaningful filter: a continuous threading coordinate such as the Gauss linking integral of the hook centreline with the loop (§5), the insertion depth along the loop axis, and/or clearance. The Reeb graph's branches give the threading-state graph, with edges at the passages.
- **Ball Mapper on hook free samples.** With ε about the typical passage width, it gives an ε-scale nerve. Components and articulation points of the ball graph mark passages, without a filter function.
- **Peg: what function to feed mode-seeking.** Mode-seeking on *sample density* finds where the sampler spent time. Instead, run ToMATo or Morse–Smale on a stratum-indicator function over the kNN pose graph, for example 1 − Euclidicity, the normal-cone rank or eigenvalue gap (VCM, §3), or HADES' p-value. The sheets are then the high-persistence plateaus and the creases are the "valleys" that separate them. This is a topological post-processing of the §2–§3 scores and replaces the fixed thresholds of GAD.
- **TPCC/TOPF are of limited use for the peg** (its sheets carry little global H1/H2). For the hook they *might* separate threaded from non-threaded regions if the free-space samples carry an H1 class around the wire (unverified). The cost is high, so use them on landmarks only.

### Gaps
- No published application of ToMATo, Morse–Smale or Mapper to C-space free-space segmentation at narrow passages was found (the closest is Pokorny & Kragic 2015).
- The kNN-graph Morse–Smale approximation (`msr`) has no consistency theory in high dimension, per the search summary.
- I did not verify the memory requirements of GUDHI cubical complexes in 6D.

## 5. Persistent homology for trajectory and homotopy classes, relevant to hook threading

### Takeaway
Pokorny and colleagues built the relevant toolkit:
- relative-persistent-homology trajectory clustering, linear in the number of trajectories (ICRA 2016, Pokorny–**Goldberg**–Kragic; the brief's "Pokorny, Hawasly, Stork, Kragic" conflates several papers);
- multiscale trajectory classification (RSS 2014);
- persistent cohomology on free-space samples (RSS 2015);
- the **Gauss linking integral** as a topological measure of caging and threading for objects with holes and for hooking grasps (ICRA 2013, HUMANOIDS 2013).

For the hook, the linking integral of the hook centreline with the wire loop is a continuous, label-free threading coordinate. The user's random-walk trajectories can be classified by the relative homology class they trace around the loop.

### Cited Findings
**Trajectory clustering and classification**
- *Pokorny, Goldberg, Kragic, "Topological trajectory clustering with relative persistent homology" (ICRA 2016).*
  - Clusters trajectories by relative persistent homology. For a fixed simplicial representation and trajectory discretization it needs only linear time in the number of trajectories, versus quadratic time for a pairwise-distance (e.g. Fréchet) matrix, and it naturally separates topologically distinct trajectories.
  - Tested on car GPS, ship and pedestrian data: 177 car trajectories at an intersection gave 24 classes.
  - [Berkeley PDF](https://goldberg.berkeley.edu/pubs/icra2016-final-topo-traj-clustering.pdf); [KTH PDF](https://www.csc.kth.se/~fpokorny/static/publications/pokorny2016b.pdf)
- *Multiscale topological trajectory classification with persistent homology (RSS 2014).* — [RSS X p54](https://roboticsproceedings.org/rss10/p54.html); [Edinburgh PDF](https://rad.inf.ed.ac.uk/data/publications/2014/rss14_majd.pdf). The author list (Pokorny, Hawasly, Ramamoorthy) is BK.
- *Winding-augmented motion planning (ICRA 2016).* Uses 2D topological task projections and persistent homology. — [PDF](https://goldberg.berkeley.edu/pubs/icra2016-final-winding-augmented-planning.pdf)
- *Data-driven topological motion planning with persistent cohomology (RSS 2015).* Free-space samples only, planning within sub- or superlevel sets, "scales well in low dimensions." — [RSS XI p49](https://roboticsproceedings.org/rss11/p49.html)

**Linking-number methods for holes, caging and hooking**
- *Pokorny, Stork, Kragic, "Grasping objects with holes: a topological approach" (ICRA 2013).* Builds a simplicial complex from a noisy cloud, finds graspable loops with shortest homology generators, and uses the **Gauss linking integral** as evidence that a caging grasp is secure. — [KTH PDF](https://www.csc.kth.se/~fpokorny/static/publications/pokorny2013a.pdf)
- *Stork, Pokorny, Kragic (HUMANOIDS 2013).* A loop-based topological representation for **clasping, latching and hooking** grasps. — [KTH PDF](https://www.csc.kth.se/~fpokorny/static/publications/stork2013c.pdf)
- *Marzinotto et al. (2014).* Extends the linking-integral idea to cooperative hooking grasps with two NAO robots. — [PDF](https://www.csc.kth.se/~almc/pdf/cooperative_topological.pdf)
- *Related writhe/interaction-mesh representations (Toussaint/Komura groups).* (BK) [Zarubin et al. ICRA workshop 2013](https://argmin.lis.tu-berlin.de/papers/13-zarubin-ICRAws.pdf)

### Inferences
- **Threading coordinate for the hook.** For a closed loop L and an *open* hook curve γ, the Gauss linking integral equals (Ω_L(γ end) − Ω_L(γ start))/4π plus integer contributions, where Ω_L is the solid angle subtended by L. It varies continuously and increments by about 1 each time the hook passes through the loop's spanning disk (standard result; BK).
  - *Use.* Computed per pose from the known hook centreline and loop wire, it gives a scalar "threading depth" without labels. Use it as the Mapper/Reeb filter (§4), as an extra coordinate for spectral clustering, or as the function in ToMATo.
  - *Expected effect.* It should separate the threading states that pose-only spectral clustering mixes, because those states differ in this homotopy-type invariant even when the poses are metrically close.
  - *Status.* This is an inference: the cited caging/hooking papers use the linking integral for grasp evidence, not for mode segmentation.
- **Classifying random-walk trajectories.**
  1. Build a simplicial complex of free space, for example a Rips or alpha complex on landmarks of the free grid nodes, or a cubical complex on {sep > c}.
  2. Discretize each trajectory segment as a chain.
  3. Classify each segment by its relative homology class with respect to the loop obstacle, following Pokorny–Goldberg–Kragic.
  4. The classes are threading histories, and transitions between classes locate the passages.
  - *Cost.* Linear in the number of trajectories once the complex exists.
  - *Caveat.* The complex must be built in a low-dimensional projection, such as hook-tip position (3D) or tip + axis, as Pokorny et al. do with 2D task projections. Full 6D complexes are expensive (inference).
- **Persistent cohomology of the clearance superlevel filtration** {sep > c} on free-space samples (RSS 2015 style) shows whether the free space has persistent H1 classes, such as cycles around the wire. The clearance thresholds at which those classes are born or die are the passage widths, which complements the 0-dimensional merge tree of §4.
- **For the peg,** trajectory homology is of little use (the contact surface is simply connected locally). The useful trajectory signal there is *where* random walks cross creases, i.e. the edges whose endpoints have different normal-cone rank or Gauss-map cluster count (§3).

### Gaps
- No TDA paper on hook-in-eyelet or threading-state segmentation of C-space was found.
- Whether the hook's 6D free space has nontrivial persistent H1 at the relevant clearance is unknown and needs computing.
- I did not find code for relative-persistent-homology trajectory clustering.

### Catalogue matrix (summary across all five questions; "Scales" means plausible on 1e5+ points in 4–6D on a workstation, by the inference in each section)

| Method | Detects | Code (maintained?) | Scales 1e5+ in 4–6D? | Peg use | Hook use |
|---|---|---|---|---|---|
| Local homology transfer (BWM 2012), Skraba–Wang | stratum equivalence classes | none found | No (Delaunay; toy grids only) | theory only; misses wedge creases | — |
| Nanda / Asai–Shah canonical stratification | coarsest cohomology-manifold strata of a complex | pseudocode only | No (needs a clean complex) | — | — |
| GAD (Stolz et al. 2020) | boundary / manifold / intersection via annulus H_{k−1} bars | MATLAB + Ripser, GitHub (research code) | Landmarks only (hours for 6k pts) | boundaries, ≥3-sheet junctions; misses 2-sheet wedges | passages / boundary of free samples |
| TARDIS Euclidicity + PID (ICML 2023) | multi-scale non-Euclidean score; local dimension | Python, BSD-3, GitHub ("work in progress") | Landmarks only (cluster for large examples) | PID for positive-mass lower strata; weak on wedges | necks |
| HADES (arXiv 2023; SIAM journal) | non-disk-like neighbourhoods (MMD vs uniform disk) | no public repo found; easy to reimplement | Yes (O(k²D) per point) | creases likely; needs density equalization | necks |
| Tangent blow-up (arXiv 2026) | separation of sheets by tangent (level 1) and curvature (level 2) | none found | Yes (≤200k tested) | = the user's normal feature; add level 2 for C¹ boundaries | — |
| Local-PCA spectral (Arias-Castro et al. 2017) | intersecting-manifold clustering | (code not checked) | Yes with sparse kNN | same as spectral-normal; theory for intersections | — |
| VCM / δ-VCM (Monte-Carlo) | normal cone (rank = number of contacts), sharp features | CGAL (3D only); Monte-Carlo variant easy in any dimension | Yes | **normal-cone rank = contact codimension** | gradient-discontinuity bisectors |
| Gauss-map clustering | number of normal clusters (sheet/crease/corner) | none; trivial to implement | Yes | crease/corner count | — |
| Hough normals (Boulch–Marlet) | robust normals at edges | C++ header-only (3D) | 3D yes; 4D needs reimplementation | robust normals for the lift | — |
| Jets / DoN / RIMLS / Fleishman | curvature; multi-scale normal change; feature-preserving fits | CGAL/PCL (3D) | 4D needs reimplementation | curvature jumps at tangent-continuous boundaries | — |
| PIE-Net / EC-Net / NerVE / DEF | 3D CAD edges | yes (3D, learned) | N/A | not applicable | not applicable |
| Calder–Park–Slepčev boundary | boundary normal / distance / strip | unknown | Yes (kNN statistics) | sheet ends, acute wedges | free-space boundary |
| ToMATo (GUDHI) | prominence-based basins of any function on any graph | GUDHI (maintained) | Yes (≤2e9 points) | on a stratum-indicator function, not on density | **passages = merge-tree saddles of clearance** |
| Morse–Smale (msr, Morse crystals; CGW theory) | ascending/descending cells | R `msr` | Moderate | — | chambers (ascending), closest-feature modes (descending) |
| Mapper / Ball Mapper | nerve / mode graph | pyBallMapper, R BallMapper, KeplerMapper/giotto (BK) | Yes | mode graph from strata | threading graph with a linking-integral filter |
| TPCC / TOPF | points by role in global homology | RWTH GitLab / pip `topf` | TPCC no; TOPF moderate | low | maybe (H1 around the wire) |
| Relative-PH trajectory clustering; persistent cohomology planning (Pokorny) | homotopy/homology classes of trajectories and free space | not found | Linear in trajectories; complex built in a low-dimensional projection | low | **threading histories of random walks** |
| Gauss linking integral | continuous threading measure | trivial to implement | Yes | — | **threading coordinate** |
