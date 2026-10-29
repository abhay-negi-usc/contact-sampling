# Multi-manifold, hybrid/subspace and stratified-space clustering methods for unsupervised contact-mode discovery

Scope note for the report writer: what follows is a catalogue of methods grouped by the seven key questions, plus an eighth section that maps methods to the specific failure modes of the peg-in-hole / hook-in-eyelet data. Every method entry gives the core idea, assumptions, inputs, guarantees, scale, code and fit to the contact problem. Facts carry inline sources. Statements about how a method would behave on the user's data are my inferences and sit under "Inferences". The web-search budget ran out near the end of the session, so a few code-availability and maintenance questions are listed under "Gaps" rather than checked.

Problem recap used for the fit notes (from the repo docstrings, not a web source). Peg-in-hole samples lie in a 4D ambient space (rho, radial tilt, tangential tilt, depth) with the tilts scaled by a lever L. Single-contact modes are 3D sheets (codimension 1) and two-contact modes are 2D creases (codimension 2). Pose is continuous across mode boundaries, some boundaries are tangent-continuous, and density is very non-uniform. The hook-in-eyelet modes are full-dimensional (6D) free-space regions separated by narrow passages.

---

## Q1. Spectral methods built on local tangent / covariance similarity, classic and current (2020–2026)

### Takeaway
The tangent-aware spectral family covers the user's crease problem best. It includes Goldberg et al. 2009, SMMC 2011, RMMSL 2012, Arias-Castro–Lerman–Zhang 2017, Deutsch–Medioni 2015, the Trillos–He–Li theory (JMLR 2023) and LAPD (2025). Each method cuts where neighbouring tangent spaces disagree, so it separates creases (tangent jumps) and cannot separate tangent-continuous boundaries. Most assume equal intrinsic dimension and a known number of clusters K. Only the Trillos–He–Li theory explicitly allows differing dimensions, and only LAPD (by persistence) and the LLPD-style eigengap give a principled way to choose K.

### Cited Findings

**Goldberg et al., "Multi-manifold semi-supervised learning" (AISTATS 2009)**
- Builds a graph over local Gaussian (covariance) fits and compares them with a Hellinger-type affinity, explicitly to handle manifolds that may differ in dimensionality, orientation and density. Arias-Castro et al. call it the main inspiration for their own method, describing it as "ℓ-nearest-neighbor centers, Mahalanobis distances, and Hellinger affinities, with constrained K-means", with only empirical evaluation. — [Goldberg et al. PMLR v5](https://proceedings.mlr.press/v5/goldberg09a/goldberg09a.pdf); [Arias-Castro et al. arXiv 1301.2007](https://ar5iv.labs.arxiv.org/html/1301.2007)

**SMMC: Spectral Multi-Manifold Clustering (Wang, Jiang, Wu, Zhou)**
- Venue correction: the paper is *IEEE Transactions on Neural Networks* 22(7):1149–1161 (2011), not JMLR. A closely related IJCAI 2011 paper is "Local and Structural Consistency for Multi-Manifold Clustering". — [SMMC GitHub implementation README](https://github.com/johnsmith0031/SMMC); [LAMDA PDF "tnn11smmc"](https://www.lamda.nju.edu.cn/publication/tnn11smmc.pdf); [IJCAI 2011 abstract](https://ijcai.org/Abstract/11/262)
- Method as summarised by Arias-Castro et al.: it fits a mixture of affine subspaces (MPPCA) to estimate tangent spaces, and the affinity is a product of cosines of the principal angles between tangent spaces (times a Euclidean kNN term). It assumes equal dimensions and has no theoretical guarantees. — [Arias-Castro et al.](https://ar5iv.labs.arxiv.org/html/1301.2007)
- Code: a third-party GitHub implementation exists (johnsmith0031/SMMC). — [GitHub](https://github.com/johnsmith0031/SMMC)

**Spectral clustering based on local PCA (Arias-Castro, Lerman, Zhang, JMLR 18(9):1–57, 2017; arXiv 1301.2007)**
- Algorithm (their Algorithm 4):
  - Pick an r-packing of centres by greedy random selection.
  - Run local PCA in r-balls and form the projection Q_i onto the top-d eigenvectors.
  - Affinity W_ij = exp(−‖y_i−y_j‖²/ε²)·exp(−‖Q_i−Q_j‖²/η²).
  - Run Ng–Jordan–Weiss spectral clustering on the centres, then give each point its nearest centre's label.
  - ε and η are set automatically: ε is the largest nearest-centre distance, η the median projection distance among ε-close pairs.
  - [ar5iv 1301.2007](https://ar5iv.labs.arxiv.org/html/1301.2007); [JMLR page](https://jmlr.org/papers/v18/14-318.html)
- Assumptions: K and d are known, and all surfaces share dimension d (Algorithm 4 needs a single d). Intersections are allowed at a strictly positive incidence angle. Surfaces are C², compact, of positive reach, with an intersection set of positive reach. Boundaries are handled only empirically. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1301.2007)
- Guarantees (Theorem 1, for the simpler threshold and connected-component variants, Algorithms 2–3, with K=2):
  - With probability ≥ 1 − C·n·exp(−n r^d η²/C), points from different surfaces are merged only if they lie within C·r of the intersection.
  - The spectral version (Algorithm 4) has no proof. The authors note that "intersection points may form their own cluster" and could not rule this out.
  - [ar5iv](https://ar5iv.labs.arxiv.org/html/1301.2007)
- Mixed dimension: the analysis assumes equal dimensions. The authors say clusters of different dimension can be separated given accurate local dimension estimates (citing Haro et al.). — [ar5iv](https://ar5iv.labs.arxiv.org/html/1301.2007)
- K selection: K is assumed known, and no practical estimator is provided. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1301.2007)
- Scale: no complexity bound is given. The r-packing subsampling reduces the spectral problem to the number of centres. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1301.2007)
- Code: the paper links http://www.ima.umn.edu/~zhang620/ (an old IMA page; I did not check whether it is live). — [ar5iv](https://ar5iv.labs.arxiv.org/html/1301.2007)

**RMMSL: Robust Multiple Manifolds Structure Learning (Gong, Zhao, Medioni, ICML 2012)**
- Two stages:
  1. Estimate local tangent spaces by weighted low-rank matrix factorization.
  2. Cluster with a "curved-level similarity" designed to favour the flattest possible clusters.
  - Evaluated on synthetic data, digits, motion capture and video. — [ICML PDF](https://icml.cc/imls/conferences/2012/papers/191.pdf); [arXiv 1206.4624](https://arxiv.org/abs/1206.4624)
- Per Arias-Castro et al., it uses robust local PCA with dimension chosen by the largest spectral gap and a self-tuning affinity, with no theoretical guarantees. — [ar5iv 1301.2007](https://ar5iv.labs.arxiv.org/html/1301.2007)

**Intersecting manifolds: detection, segmentation and labeling (Deutsch & Medioni, IJCAI 2015)**
- Pipeline:
  1. Tensor voting gives per-point tangent and normal spaces.
  2. Intersection areas are flagged by peaks of the "ball" eigenvalue λ_N of the voted tensor (threshold λ_N > 2σ), or by a large λ_{d+1}, and removed.
  3. The smooth parts are spectrally clustered on a Tensor Voting Graph whose affinity is a local tangent-space distance.
  4. A greedy "ambiguity resolution" relabels the junction points by minimising total variation of tangent spaces.
  - [IJCAI 2015 PDF](https://www.ijcai.org/Proceedings/15/Papers/485.pdf)
- Key empirical statement: earlier tangent-space affinity methods (Wang 2011, Gong 2012, Goldberg 2009, Arias-Castro) "only provide satisfactory results when the angle between the tangent planes is large (typically > π/4)". The authors also found tangent estimation more accurate with tensor voting than with local PCA. Cost: the greedy step is O(j·N·k log k), with j junction-area points and N the ambient dimension. — [IJCAI 2015 PDF](https://www.ijcai.org/Proceedings/15/Papers/485.pdf)
- They cite Belkin et al. 2012 on how disruptive the few near-intersection points are to global structure estimation. — [IJCAI 2015 PDF](https://www.ijcai.org/Proceedings/15/Papers/485.pdf)

**Why plain kNN-graph spectral clustering misbehaves at creases (Belkin, Que, Wang, Zhou, COLT 2012)**
- Near intersections and sharp "edges", the graph Laplacian does not converge to Laplace–Beltrami. It tends to a first-order differential operator with different kernel-width scaling and shows a Gibbs-like effect. Points near singularities are a small volume fraction but contribute disproportionately. — [PMLR v23](https://proceedings.mlr.press/v23/belkin12.html); [arXiv 1211.6727](https://arxiv.org/abs/1211.6727)

**Large-sample spectral analysis of graph-based multi-manifold clustering (García Trillos, He, Li, JMLR 24, 2023; arXiv 2107.13610)**
- Setting: a union of manifolds that may intersect and may have *different dimensions*.
- The paper states conditions under which a similarity graph's Laplacian captures the geometry needed for MMC, with "annular proximity graphs with angle constraints" as an example family, contrasted with tangent-plane-alignment constructions.
- It proves high-probability error bounds for the graph Laplacian approximating a "tensorized Laplacian" on the union, which contains each manifold's geometry separately. The abstract says nothing about choosing K.
- [arXiv 2107.13610](https://arxiv.org/abs/2107.13610); [JMLR v24 PDF](https://jmlr.org/papers/volume24/21-1254/21-1254.pdf)
- The same paper observes that Singer & Wu's connection-Laplacian methodology "can be considered as a MMC algorithm since it also uses tangent plane information". It also notes (as summarised by search) that SCC works on linear manifolds but fails with curvature. — [JMLR v24 PDF](https://jmlr.org/papers/volume24/21-1254/21-1254.pdf)

**LAPD: Robust Multi-Manifold Clustering via Simplex Paths (Chen, Little, Narayan, arXiv 2507.10710, 2025; conference version 2023)**
- Algorithm:
  - Build candidate d-simplices from an annular kNN graph, filtered by size and shape distortion.
  - Weight adjacent simplices by dihedral angle.
  - Define the *largest angle path distance*: a path costs its sharpest turn, and the distance is the minimax over paths.
  - Denoise simplices by their κ-th-NN LAPD, then cluster by multiscale single linkage (from Little et al. 2020).
  - Choose the number of clusters by persistence and vote labels back to points.
  - [ar5iv 2507.10710](https://ar5iv.labs.arxiv.org/html/2507.10710)
- Assumptions: all manifolds share intrinsic dimension d, and intersections have dimension d−1 at angle Θ. The data come from a noisy tube of radius τ.
- Theory: within-manifold LAPD ≲ τ/e and between-manifold LAPD ≳ Θ, with high probability (analysis for two linear manifolds). — [ar5iv](https://ar5iv.labs.arxiv.org/html/2507.10710)
- Scale: quasi-linear in n, linear in ambient D, exponential in d ("impractical when d > 5"). Runtime experiments go up to n = 35,000. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2507.10710)
- Code: https://github.com/HYfromLA/LAPD. On the authors' benchmarks it is competitive with or better than SSC-ADMM, EKSS, LocPCA (Arias-Castro), PBC and DCV, with SSC-ADMM and EKSS best on pure subspace tasks. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2507.10710)

**Path-constrained Isomap-style relatives**
- "Multiple Manifold Clustering Using Curvature Constrained Path" (Babaeian et al., PLOS One 2015) — [PLOS One](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0137986); [arXiv 1812.02327](https://arxiv.org/pdf/1812.02327)
- "Angle constrained path to cluster multiple manifolds" (a constrained Isomap variant) — [arXiv 1802.07416](https://arxiv.org/pdf/1802.07416)
- The curvature-constrained-path paper notes that most methods are designed for non-intersecting surfaces. — [arXiv 1812.02327](https://arxiv.org/pdf/1812.02327)

**Spectral Curvature Clustering (SCC; Chen & Lerman) and Kernel SCC**
- SCC combines Govindu's multi-way spectral framework with Ng et al., using a polar-curvature affinity on (d+2)-tuples, for mixtures of affine flats. The practical IJCV algorithm needs linear storage and linear time in n. — [Chen thesis](https://pdp.sjsu.edu/faculty/guangliang.chen/papers/thesis_gchen.pdf); [IJCV record](https://datalearner.com/academic/journal-papers/0920-5691/volumes-and-issues/81/paper-detail/13501); [arXiv 0810.3724](https://ar5iv.arxiv.org/html/0810.3724)
- The "Foundations" theory assumes equal-dimension flats for simplicity and notes that it extends to mixed dimension by using the maximum dimension. — [Projecteuclid EJS](https://projecteuclid.org/download/pdfview_1/euclid.ejs/1322057436)
- Kernel SCC (ICCV Workshops 2009) handles nonlinear surfaces through kernels. — [mlanthology KSCC](https://mlanthology.org/iccvw/2009/chen2009iccvw-kernel)

**Survey**
- Abdolali & Gillis, "Beyond Linear Subspace Clustering: A Comparative Study of Nonlinear Manifold Clustering Algorithms" (Computer Science Review 42, 2021). It groups methods into locality-preserving, kernel-based and neural-network-based categories and compares representatives on synthetic and real data. — [arXiv 2103.10656](https://arxiv.org/abs/2103.10656v1)

### Inferences
- **Crease fit.** Every method in this family turns a jump in tangent space into a weak edge. On the peg's 4D data the tangent space of a 3D sheet is the orthogonal complement of a single normal. So "projection distance ‖Q_i−Q_j‖" (Arias-Castro), "product of cosines of principal angles" (SMMC) and the user's "normal agreement" are the same signal up to a monotone map. Gains over what the user already tried must therefore come from:
  1. better tangent estimation at creases (tensor voting, robust or weighted low-rank fits, multiscale or diffusion-geometry estimates);
  2. explicit removal and later relabelling of junction samples (Deutsch–Medioni);
  3. replacing the summed-affinity spectral objective with a minimax path criterion (LAPD) or an angle-constrained annular graph (Trillos–He–Li).

  Options 2 and 3 are the parts the user has not yet tried.
- **Junction clusters are wanted here.** Arias-Castro et al. list it as a flaw that "intersection points may form their own cluster". For contact modes that behaviour is desirable, because the crease *is* a two-contact mode. Running their Algorithm 4 with K larger than the number of sheets may therefore produce the two-contact modes for free.
- **LAPD matches the peg geometry closely.** Its model (same d, (d−1)-dimensional intersections) is exactly 3D sheets meeting along 2D creases, d=3 is inside its practical range (d ≤ 5), and it picks K by persistence.
  - The theory covers *crossing* flats, not *folded* sheets that end at the crease. A path between two folded half-sheets still has to turn by the dihedral angle, so the separation argument should carry over, but this is not proven.
  - Points sampled *on* the crease form flat (degenerate) 3-simplices that the shape-distortion filter would discard. They would then be voted into a neighbouring sheet, so two-contact modes would not come out as clusters. They need to be peeled off first by a dimension test (Q3).
  - Scale: it was tested to 35k points, so the user's 1e6 samples would need thinning to roughly 1e4–5e4 (for example an r-packing or the user's existing "even" grid sampling).
- **Tangent-continuous boundaries are out of reach for this family.** Θ = 0 there, so LAPD's between-cluster distance bound ≳ Θ vanishes, and every tangent-affinity method sees one smooth sheet. This is a property of the geometry, not a tuning problem. It is consistent with the user's supervised exact-mode ceiling (0.62–0.75) being far below the family ceiling (~0.99).
- **Shallow creases may be invisible.** Deutsch–Medioni's "> π/4" observation suggests that creases with small dihedral angles will be missed by tangent-affinity methods even when they are true creases. The user could tabulate the analytic dihedral angle of each peg crease to predict which mode boundaries are recoverable at all.

### Gaps
- I could not confirm whether the Arias-Castro/Zhang IMA code page is still live, or whether a maintained Python port exists.
- I found no 2024–2026 *spectral* MMC method other than LAPD (2025) and the Trillos–He–Li theory. A targeted venue search (NeurIPS/ICML/ICLR 2024–2026) was not possible after the search budget ran out.
- No source gives the exact theorem conditions in Trillos–He–Li for intersections of *different* dimensions; only the abstract was read.

---

## Q2. Sparse / low-rank self-expressive methods (SSC, LRR, SMCE, deep subspace clustering): usable for nonlinear, mixed-dimensional, intersecting manifolds?

### Takeaway
Global self-expressive methods (SSC/LRR) are provably good for unions of *linear* subspaces, including intersecting ones of different dimension, and scale to ~1e5–5e5 points (SSC-OMP/EnSC). They are a poor match for curved 3D sheets in a 4D ambient space, where codimension 1 leaves almost no room for subspace-preserving sparse codes. SMCE (the local, manifold version) estimates local dimension but failed on intersecting manifolds in head-to-head tests. Self-expressive deep subspace clustering has a documented ill-posedness problem.

### Cited Findings

**SSC theory**
- Soltanolkotabi & Candès (Ann. Statist. 40(4):2195–2238, 2012) analyse SSC for points near a union of lower-dimensional planes. They do not assume the number or dimensions of the subspaces are known. They show SSC recovers subspaces of dimension comparable to the ambient dimension, clusters correctly even when subspaces *intersect*, and has an extension robust to overwhelmingly many outliers. — [arXiv 1112.4258](https://arxiv.org/abs/1112.4258v5); [Projecteuclid](https://projecteuclid.org/journals/annals-of-statistics/volume-40/issue-4/A-geometric-analysis-of-subspace-clustering-with-outliers/10.1214/12-AOS1034.pdf)
- The noisy-data version is "Robust Subspace Clustering" (Soltanolkotabi, Elhamifar, Candès, Ann. Statist. 42(2):669–699, 2014). — [Projecteuclid](https://projecteuclid.org/journals/annals-of-statistics/volume-42/issue-2/Robust-subspace-clustering/10.1214/13-AOS1199.full)
- The core algorithm paper is Elhamifar & Vidal, "Sparse Subspace Clustering: Algorithm, Theory, and Applications". — [arXiv 1203.1005](https://arxiv.org/abs/1203.1005)

**Scaling SSC**
- SSC-OMP (You, Robinson, Vidal, CVPR 2016) is computationally efficient and guarantees subspace-preserving affinities under broad conditions. It was "the first one to handle 100,000 data points", slightly less accurate than SSC-BP but orders of magnitude faster. — [arXiv 1507.01238](https://arxiv.org/pdf/1507.01238)
- The companion elastic-net SSC (EnSC) uses an L1+L2 penalty for scalability and graph connectivity, with a provably correct active-set solver. — [arXiv 1507.01238](https://ar5iv.arxiv.org/html/1507.01238)
- Maintained code: github.com/ChongYou/subspace-clustering. It is Python with a scikit-learn-style API (`fit`, `labels_`) for EnSC and SSC-OMP, with synthetic experiments from 500 to 0.5 million points and MNIST (70k) "within about an hour". — [GitHub](https://github.com/ChongYou/subspace-clustering)

**LRR**
- LRR (Liu et al.) is the low-rank analogue and is cited by LAPD as a baseline family. I did not retrieve its own paper. — [LAPD related work, ar5iv 2507.10710](https://ar5iv.labs.arxiv.org/html/2507.10710)

**SMCE: Sparse Manifold Clustering and Embedding (Elhamifar & Vidal, NIPS 2011)**
- Each point is sparsely reconstructed from a locally weighted neighbourhood. The number of neighbours selected varies per point and serves as an estimate of local manifold dimension, motivated by data on "multiple manifolds of possibly different dimensions". MATLAB code is on the author's site. — [NeurIPS 2011](https://proceedings.neurips.cc/paper/2011/hash/fc490ca45c00b1249bbe3554a4fdf6fb-Abstract.html); [paper PDF](https://www.cis.jhu.edu/~ehsan/Downloads/SMCE-NIPS11-Ehsan.pdf); [code page](https://www.cis.jhu.edu/~ehsan/code.htm)
- Independent test (Allegra et al. 2020), with K fixed to the truth and the best of a parameter sweep:
  - SMCE does very well on two well-separated manifolds (NMI ≥ 0.94).
  - It "cannot correctly retrieve intersecting manifolds". On five Gaussians of dims 1, 2, 4, 5, 9 NMI was 0.73 (linear) and 0.61 (curved), versus Hidalgo's 0.89 and 0.84, and its ID estimates were poor.
  - SMCE requires K as input.
  - [Allegra et al., Sci. Rep. 2020](https://doi.org/10.1038/s41598-020-72222-0) (full text read from [UniCatt PDF](https://publicatt.unicatt.it/retrieve/e309db6f-8864-0599-e053-3705fe0a55db/SCIREP_Hidalgo.pdf))

**Deep self-expressive / manifold-linearizing methods**
- Haeffele, You & Vidal (ICLR 2021) show that self-expressive deep subspace clustering (SEDSC) objectives are often ill-posed, admitting degenerate embeddings. Re-running prior experiments, they found that much of the reported gain came from ad-hoc post-processing. — [arXiv 2010.03697](https://arxiv.org/abs/2010.03697v2); [ICLR 2021](https://iclr.cc/virtual/2021/poster/2688)
- MLC (Ding, Tong, Chan, Dai, Ma, Haeffele, ICCV 2023) jointly learns a linearising representation and a doubly stochastic membership by optimising Maximal Coding Rate Reduction. It targets image datasets (CIFAR-10/100, TinyImageNet). — [ICCV 2023 PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Ding_Unsupervised_Manifold_Linearizing_and_Clustering_ICCV_2023_paper.pdf); [arXiv 2301.01805](https://arxiv.org/abs/2301.01805v2)
- Abdolali & Gillis (2021) compare kernel and neural-network nonlinear subspace clustering against locality-preserving methods. — [arXiv 2103.10656](https://arxiv.org/abs/2103.10656v1)

### Inferences
- **Codimension 1 is the hard case for self-expression.**
  - In R⁴ any 4 affinely independent points span a 3-flat. A point on sheet A can be written exactly with 4 points from A, but almost as sparsely with 5 points from any mix of sheets.
  - The margin that makes SSC "subspace-preserving" (dimension well below ambient) is therefore tiny, and noise or curvature will produce cross-sheet coefficients.
  - The sheets are also affine and curved, not linear through the origin, so global SSC/LRR is mismatched on two counts.
  - I would not expect SSC/LRR on raw 4D poses to beat the user's normal-based spectral results.
- **Wrench features raise the codimension.** Concatenating wrench features, which the user found raised family ARI to ~0.85, lifts the ambient dimension while the per-mode manifold stays ~3D (wrench is approximately a function of pose and slip direction). That increases codimension and makes *local* self-expression (SMCE-style, or EnSC on kNN-restricted dictionaries) more viable. The joint (pose, wrench) space is where SSC-OMP/EnSC with their maintained Python code would be worth a quick try at 1e5 points.
- **Deep methods are low priority.** Deep subspace / MCR² methods are designed for high-dimensional images, and given the ICLR 2021 critique they do not address the core issue (tangent-continuous boundaries).

### Gaps
- I did not retrieve the LRR primary paper (Liu et al., TPAMI) or any LRR result on intersecting curved manifolds.
- No source found for SSC behaviour specifically at codimension 1; the margin argument above is my own.

---

## Q3. Intrinsic-dimension-based segmentation and stratification learning (Hidalgo, TPMM, local ID estimators, DADApy, topological methods): can they separate a 3D sheet from a 2D crease in 4D?

### Takeaway
Yes, in principle, *if* the 2D crease carries positive sample mass (the sampler actually puts many samples exactly on two-contact configurations). Hidalgo separated manifolds whose IDs differed by 1 (1 vs 2, 4 vs 5) at NMI 0.84–0.89 with intersections present, and it chooses K by the posterior. It scales roughly quadratically in N in practice, so it needs ~1e4–1e5-point subsamples. If crease samples are only sheet samples *near* the crease, local ID stays 3 and ID methods cannot see the crease. Topological methods (local homology, Euclidicity) detect only *topological* singularities, so they also will not flag a geometric fold of a single 3-manifold.

### Cited Findings

**Hidalgo (Allegra, Facco, Denti, Laio, Mira, Sci. Rep. 10:16449, 2020)**
- Model: a Bayesian mixture of K Pareto laws on μ = r₂/r₁, the ratio of second- to first-neighbour distances, whose law depends only on the local ID (TwoNN). — [Sci. Rep. 2020 (UniCatt PDF)](https://publicatt.unicatt.it/retrieve/e309db6f-8864-0599-e053-3705fe0a55db/SCIREP_Hidalgo.pdf)
- Neighbourhood-uniformity prior: the q nearest neighbours of a point share its label with probability ξ, and the posterior is sampled by Gibbs sampling. Without the prior (ξ = 0.5) it "is not able to correctly separate the points" (NMI < 0.001). The recommended setting is q = 3, ξ = 0.8. — [UniCatt PDF](https://publicatt.unicatt.it/retrieve/e309db6f-8864-0599-e053-3705fe0a55db/SCIREP_Hidalgo.pdf)
- Density robustness: built on TwoNN, which "is insensitive to density variations and uses only the distances between points". Uniformity is required only up to the second neighbour, and the authors stress it works with non-uniform densities and curved, intersecting manifolds. — [UniCatt PDF](https://publicatt.unicatt.it/retrieve/e309db6f-8864-0599-e053-3705fe0a55db/SCIREP_Hidalgo.pdf)
- Results (N = 1000 points per manifold, partly overlapping):
  - Two Gaussians, d = 4 vs 5–9: NMI ≈ 0.91–0.94.
  - Five *intersecting* Gaussians of dims 1, 2, 4, 5, 9: NMI 0.89, estimated dims 0.9/1.9/4.2/4.6/8.9.
  - Five curved, intersecting manifolds (including Swiss roll and 3-torus): NMI 0.84, with some 3-torus points misassigned to a 4D Swiss roll.
  - [UniCatt PDF](https://publicatt.unicatt.it/retrieve/e309db6f-8864-0599-e053-3705fe0a55db/SCIREP_Hidalgo.pdf)
- K selection: fit K = 1, 2, … and stop when the average log-posterior stops improving. It recovered K = 5 in both synthetic tests. — [UniCatt PDF](https://publicatt.unicatt.it/retrieve/e309db6f-8864-0599-e053-3705fe0a55db/SCIREP_Hidalgo.pdf)
- Cost:
  - The Gibbs sweep is O(K·N·q²), the number of iterations was empirically ~linear in N, and so "in practice the total complexity of Hidalgo is roughly quadratic in N". Experiments used 1e5 Gibbs iterations and 10 restarts. — [UniCatt PDF](https://publicatt.unicatt.it/retrieve/e309db6f-8864-0599-e053-3705fe0a55db/SCIREP_Hidalgo.pdf)
  - The abstract nonetheless calls it "computationally efficient… even on large data sets" (real tests: 31,000 protein frames, 29,851 voxels).
  - The intRinsic paper notes that fitting "may become time consuming if the analyzed datasets are large" and that Hidalgo lacks an EM/frequentist counterpart. — [intRinsic arXiv 2102.11425](https://arxiv.org/pdf/2102.11425)
- Limitation stated by the authors: ID segmentation is not a substitute for clustering. Clusters with the same ID "cannot be identified as different manifolds by Hidalgo". — [UniCatt PDF](https://publicatt.unicatt.it/retrieve/e309db6f-8864-0599-e053-3705fe0a55db/SCIREP_Hidalgo.pdf)
- Code:
  - Original repo github.com/micheleallegra/Hidalgo (MATLAB and Python folders, 30 commits, no releases). — [GitHub](https://github.com/micheleallegra/Hidalgo)
  - R package intRinsic (TWO-NN, Gride, Hidalgo; C++ Gibbs sampler via Rcpp; MIT licence; maintained by F. Denti). — [intRinsic arXiv](https://arxiv.org/pdf/2102.11425); [CRAN refman](https://ftp.fau.de/cran/web/packages/intRinsic/refman/intRinsic.html)

**Comparator: Carter et al. local ID (LID) with smoothing (from Allegra et al. 2020)**
- With k = 50 it works on two separated manifolds (NMI ≥ 0.92, except 4 vs 5) but systematically underestimates higher IDs because it assumes uniform density over large neighbourhoods.
- In the 5-manifold tests it merged the 4D and 5D manifolds, and points of the 1D manifold intersecting them contaminated the result. — [UniCatt PDF](https://publicatt.unicatt.it/retrieve/e309db6f-8864-0599-e053-3705fe0a55db/SCIREP_Hidalgo.pdf)

**DADApy (Glielmo et al., Patterns 2022)**
- Python, Apache-2.0, `pip install dadapy`, actively developed (~1,100 commits; Python 3.10–3.14).
- ID estimators: 2NN, GRIDE, I3D and BID. Density estimators: kNN, k*NN, PAk and BMTI. Density-peak and Advanced Density Peaks clustering, plus information imbalance.
- *Hidalgo is not listed* on the README or docs index.
- [GitHub](https://github.com/sissa-data-science/DADApy); [docs](https://dadapy.readthedocs.io/); [PMC article](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9583186/)

**scikit-dimension (Bac, Mirkes, Gorban, Tyukin, Zinovyev, Entropy 2021)**
- scikit-learn API, global and local ID, benchmarked on 521 datasets. — [arXiv 2109.02596](https://arxiv.org/pdf/2109.02596); [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8534554/)
- The API lists CorrInt, DANCo, ESS, FisherS, GRIDE (n1, n2, multiscale), KNN, lPCA, MADA, MiND_ML, MLE, MOM, PH, TLE and TwoNN. — [skdim API](https://scikit-dimension.readthedocs.io/en/latest/api.html)

**GRIDE (Denti et al. 2022)**
- A generalised ratio estimator using higher-order neighbour pairs, so the ID can be read across scales. Implemented in intRinsic, DADApy and skdim. — [intRinsic arXiv](https://arxiv.org/pdf/2102.11425); [skdim API](https://scikit-dimension.readthedocs.io/en/latest/api.html)

**Translated Poisson Mixture Model (Haro, Randall, Sapiro, IJCV 80(3):358–374, 2008; NIPS 2006)**
- Local point counts in balls are modelled by translated Poisson laws, which account for noise. EM soft-clusters points by *both* dimension and density, with asymptotic theory. — [NeurIPS 2006](https://proceedings.neurips.cc/paper/2006/hash/51be2fed6c55f5aa0c16ff14c140b187-Abstract.html); [Duke record](https://scholars.duke.edu/publication/809212)
- The CVPR 2007 follow-up adds regularisation and spatial-continuity constraints. — [Princeton record](https://collaborate.princeton.edu/en/publications/regularized-mixed-dimensionality-and-density-learning-in-computer/)

**A theory of stratification learning (Aamari & Berenfeld, arXiv 2405.20066, May 2024)**
- Minimax estimation for i.i.d. samples from a stratified mixture of immersed manifolds of *different* dimensions.
- A constructive, adaptive algorithm ("ascending hierarchical co-detection") estimates the number of layers, their dimensions, point-to-layer assignment and tangent spaces at optimal dimension-specific rates, claimed "without any ambient assumption on the manifolds or their intersection configurations".
- No code linked. — [arXiv 2405.20066](https://arxiv.org/abs/2405.20066)

**Topological stratification learning**
- Bendich, Wang, Mukherjee, "Local homology transfer and stratification learning" (SODA 2012): multi-scale stratification via kernel/cokernel persistent homology, with correctness and sample-complexity guarantees for deciding which points share a stratum. — [Duke record](https://scholars.duke.edu/publication/789717); [PDF](https://www.sci.utah.edu/~beiwang/publications/Stratification_Learning_BeiWang_2012.pdf); [arXiv 1008.3572](https://ar5iv.arxiv.org/html/1008.3572)
- Brown & Wang give a sheaf-theoretic alternative using the local homology sheaf. — [arXiv 1712.07734](https://arxiv.org/pdf/1712.07734)
- Stolz, Tanner, Harrington, Nanda, "Geometric anomaly detection in data" (PNAS 117(33), 2020): persistent (co)homology in local annuli stratifies a point cloud into pieces approximable by submanifolds of possibly different dimension. It flags singular regions even when no sample lies on the singularity (cyclo-octane: Klein bottle ∪ sphere glued along circles; self-intersections of a Henneberg surface). A talk abstract says it is less sensitive to neighbourhood size and curvature than local PCA. — [arXiv 1908.09397](https://arxiv.org/pdf/1908.09397); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7443892)
- von Rohrscheidt & Rieck, "Topological Singularity Detection at Multiple Scales" (ICML 2023; preprint "TOAST"): persistent local homology for local ID plus a multi-scale "Euclidicity" score of manifold-ness. Multi-scale scores had lower variance than single-scale ones. Code: the TARDIS library. — [PMLR v202](https://proceedings.mlr.press/v202/von-rohrscheidt23a.html); [arXiv 2210.00069](https://arxiv.org/pdf/2210.00069); [TARDIS docs](https://tardis-tda.readthedocs.io/)

**Diffusion-geometry estimators (Iolo Jones, arXiv 2411.04100, v3 Feb 2026; follow-up arXiv 2602.06006)**
- Curvature, tangent-space and dimension estimators that match the state of the art on clean dense data and do better under noise or sparsity. Tangent-space and scalar-curvature estimates need no parameter tuning. — [arXiv 2411.04100](https://arxiv.org/abs/2411.04100); [arXiv 2602.06006](https://arxiv.org/abs/2602.06006)

### Inferences
- **Positive mass on creases is the deciding condition.**
  - The contact surface of the peg is (generically) the boundary of a 4D C-obstacle, so it is a *topological* 3-manifold that is bent along creases.
  - (a) If the sampler places samples *exactly* on two-contact configurations with positive frequency, those samples form a genuine 2D sub-population and Hidalgo (or TwoNN/GRIDE + spatial smoothing) should separate them. Hidalgo has separated ID gaps of 1 at NMI ~0.9. This is likely for *concave* creases, where a press or projection step funnels configurations into the crease.
  - (b) Otherwise the crease is a measure-zero set of a 3D distribution and every ID method returns 3 there, including convex creases that samples merely cross.
  - Recommended check: the histogram of analytic labels already tells how many samples are two-contact, and a local TwoNN on those samples alone should give ≈2.
- **Random-walk samples can fake ID ≈ 1.** TwoNN's first and second neighbours are likely to be the previous and next steps of the same walk, which biases the estimate toward 1. Fixes:
  - exclude same-trajectory neighbours or thin each walk;
  - use GRIDE with larger neighbour orders (n1, n2), available in skdim, DADApy and intRinsic.
- **Haro's TPMM splits by density too.** It clusters by dimension *and* density. With the user's sampler-induced density variation this would split sheets by sampling density, so it is less suited than Hidalgo.
- **Topological methods miss bends.** Local homology, Euclidicity and Stolz et al. detect topological singularities: X-crossings, T-junctions, boundaries and changes of local homology. A C⁰ bend of one 3-manifold has the same local homology as a flat point, so these methods will not mark convex or concave creases. They *would* flag places where the contact surface is genuinely non-manifold (e.g., where three or more sheets meet, or where the sampled set has boundary). They are useful as a sanity check, not as the main crease detector.
- **Practical scale.** Run Hidalgo (q = 3, ξ = 0.8) on a 2e4–5e4 thinned subsample with K ∈ {2, 3, 4} chosen by log-posterior, then propagate the labels to all 1e6 samples by kNN vote.
- **Pipeline role.** Aamari–Berenfeld is the right theoretical framing (layers of different dimension, tangent spaces per layer), but there is no code. Its output structure (layer count, dimension, assignment, tangent spaces) is exactly the "first separate by dimension, then cluster each dimension by tangent" pipeline suggested in Q8.

### Gaps
- Hidalgo is not confirmed in DADApy; the only Python source is the original micheleallegra repo, whose maintenance status is unknown (no commit dates visible).
- skdim pointwise (`fit_pw`) support could not be confirmed from the API page.
- No source evaluates any ID estimator specifically on 3D-vs-2D separation in a 4D ambient space with codimension-1 sheets.

---

## Q4. Mixtures of local linear pieces (MPPCA, MFA, K-manifolds, local best-fit flats, SCC) and tensor voting for mixed-dimensional structure and junction detection

### Takeaway
Mixture-of-flats models can carry a different latent dimension per component and are cheap, but they tile a curved sheet into many flats and need a merge step. That is essentially the "patch over-segment + merge" the user already tried. Tensor voting stands out in this group: it gives per-point dimension (eigen-gap) and junction saliency (ball eigenvalue) in N-D, robust to outliers, varying density and intersections. Deutsch–Medioni wrap it in a detect-junction, cluster-smooth-parts, relabel-junction pipeline.

### Cited Findings

**Tensor voting in N-D (Mordohai & Medioni, JMLR 11:411–450, 2010)**
- Works in the original input space without dimensionality reduction. It gives a reliable *per-point* local dimensionality estimate (not a global one) and supports geodesic distances and interpolation for data "with varying density, outliers, perturbation and intersections". — [JMLR page](https://jmlr.org/papers/v11/mordohai10a.html); [PDF](https://jmlr2020.csail.mit.edu/papers/volume11/mordohai10a/mordohai10a.pdf)
- In the voted second-order tensor, normals are eigenvectors with large eigenvalues and tangents those with small ones. The "ball" component (eigenvalue λ_N) peaks at intersection areas, and λ_{d+1} jumps there in the smooth-versus-junction comparison. — [IJCAI 2015](https://www.ijcai.org/Proceedings/15/Papers/485.pdf)
- Tensor voting is strictly local. The Tensor Voting Graph (Deutsch & Medioni) adds global structure and handles manifolds with singularities and multiple intersecting manifolds. — [Medioni ICVSS abstract](https://icvss.dmi.unict.it/icvss2014/Abstracts/Medioni.pdf); [JMIV 2016/17 via unpaywall](https://unpaywall.org/10.1007%2FS10851-016-0684-2)

**Deutsch & Medioni IJCAI 2015** — pipeline and limits as in Q1. — [IJCAI 2015](https://www.ijcai.org/Proceedings/15/Papers/485.pdf)

**K-manifolds (Souvenir & Pless, ICML 2005)**
- An EM-style assignment of points to multiple manifolds, aimed at intersecting manifolds of mixed topology and dimension. — [Souvenir & Pless PDF](https://www.math.wustl.edu/~victor/classes/pmf/souvenir-ManifoldClusteringByMDS.pdf)
- Arias-Castro et al. call this line "very complex methods that are hard to analyze". — [ar5iv 1301.2007](https://ar5iv.labs.arxiv.org/html/1301.2007)

**Local best-fit flats / K-flats (Lerman group)**
- "Hybrid Linear Modeling via Local Best-fit Flats" (Zhang, Szlam, Wang, Lerman, IJCV 100(3), 2012): forms local best-fit affine subspaces with neighbourhood size set automatically via Jones' β² numbers, then selects a model greedily or spectrally. — [arXiv 1010.3460](https://www.arxiv.org/abs/1010.3460)
- Randomized LBF (CVPR 2010) minimises a global ℓ1 error over selected flats. — [arXiv 1005.0858](https://arxiv.org/pdf/1005.0858)
- Median K-Flats (online, robust to many outliers). — [arXiv 0909.3123](https://www.arxiv.org/pdf/0909.3123)

**MPPCA as a building block**
- MPPCA (Tipping & Bishop 1999) is used inside SMMC to estimate tangent spaces. — [Arias-Castro summary](https://ar5iv.labs.arxiv.org/html/1301.2007); [LAPD related work](https://ar5iv.labs.arxiv.org/html/2507.10710)

**SCC**
- Linear storage and time. Theory for equal-dimension flats, extended to mixed dimension via the maximum dimension. Fails on curved manifolds unless kernelised. — [thesis](https://pdp.sjsu.edu/faculty/guangliang.chen/papers/thesis_gchen.pdf); [EJS](https://projecteuclid.org/download/pdfview_1/euclid.ejs/1322057436); [JMLR v24 discussion](https://jmlr.org/papers/volume24/21-1254/21-1254.pdf)

### Inferences
- **Tensor voting is cheap here.** In a 4D ambient space a voted tensor is only 4×4, so tensor voting on kNN graphs of 1e6 points is a linear-time pass. Three outputs per sample map onto the problem:
  1. λ₁−λ₂ gap (stick saliency): a codimension-1 sheet point;
  2. λ₂−λ₃ gap: a codimension-2 crease point (2D);
  3. ball saliency λ₄: a junction or noise.

  This directly addresses "mixed dimension" and "crease detection" and is a principled replacement for the user's surface variation λ_min/Σλ. Deutsch–Medioni report that its tangents near intersections are more accurate than local PCA, which targets the user's observed failure of local-PCA normals near creases.
- **Mixtures over-segment curved sheets.** MPPCA/MFA/K-flats on 4D poses will cut each curved sheet into flats at arbitrary places. That only helps as an over-segmentation whose merge step uses tangent continuity across patch borders, which the user already tried. With per-component latent dimension q ∈ {2, 3}, BIC-selected, they could at least label crease components as q = 2.
- **K-manifolds and LBF are mainly history.** Use Arias-Castro (spectral) or LAPD instead. SCC's polar curvature on (d+2)-tuples is affine-only.

### Gaps
- No maintained Python implementation of N-D tensor voting or of the Tensor Voting Graph was found; the search budget ran out before checking GitHub.
- No maintained Python MPPCA/MFA package was verified (scikit-learn has none as far as these sources show).
- The JMIV Deutsch–Medioni TVG paper's exact title, venue details and algorithm were not retrieved; only the DOI and an abstract via unpaywall.

---

## Q5. Tangent-bundle / connection-based operators (vector diffusion maps, connection Laplacian, sheaf Laplacians, Grassmannian tangent clustering): do they help cut along creases?

### Takeaway
They formalise "compare neighbouring tangent spaces after transport". VDM approximates the connection Laplacian and parallel transport from local PCA plus orthogonal alignment, and the sheaf Laplacian generalises it. Trillos–He–Li note that connection-Laplacian methods can serve as MMC algorithms. In codimension 1 (3D sheets in 4D) they reduce to a smoothed normal-agreement graph, which the user already has. Their real extra value is a *global* consistency signal: alignment frustration or residual around cycles. Its main use would be an edge-level crease detector, not a different clustering.

### Cited Findings
- **Vector diffusion maps.** Singer & Wu (CPAM 2012) generalise diffusion maps to vector fields. Local PCA estimates tangent spaces, and orthogonal alignment between neighbouring tangent bases approximates parallel transport. The VDM matrix approximates the connection Laplacian under uniform sampling and can detect orientability. — [arXiv 1102.0075](https://arxiv.org/pdf/1102.0075); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3886882)
- **Connection Laplacian as MMC.** Trillos–He–Li state that Singer & Wu's (2017) connection-Laplacian methodology "can be considered as a MMC algorithm since it also uses tangent plane information to inform the affinity between points". — [JMLR v24 PDF](https://jmlr.org/papers/volume24/21-1254/21-1254.pdf)
- **Sheaf Laplacians.** Hansen & Ghrist ("Toward a Spectral Theory of Cellular Sheaves", J. Appl. Comput. Topology 2019) lift the graph Laplacian to a sheaf (Hodge) Laplacian, with results on interlacing, sparsification, effective resistance and synchronization. Small eigenvalues correspond to smooth, globally aligned signals. — [arXiv 1808.01513](https://arxiv.org/pdf/1808.01513); [Springer](https://link.springer.com/article/10.1007/s41468-019-00038-7); [arXiv 2512.00242](https://arxiv.org/pdf/2512.00242)
- **Sheaf-based stratification.** Brown & Wang use a local homology *sheaf* for stratification learning. — [arXiv 1712.07734](https://arxiv.org/pdf/1712.07734)
- **Grassmannian distances in practice.**
  - Arias-Castro uses the projection (chordal Grassmann) distance ‖Q_i−Q_j‖. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1301.2007)
  - SMMC uses the product of cosines of principal angles. — [ar5iv summary](https://ar5iv.labs.arxiv.org/html/1301.2007)
  - Wang–Slavakis–Lerman cluster *data that live on* a Grassmannian with tangent and geodesic directional information (Q6). — [arXiv 1410.0095](https://www.arxiv.org/abs/1410.0095)
- **No direct clustering paper found.** The VDM search found no paper applying VDM specifically to clustering intersecting manifolds. Tangent spaces are ill-defined at intersections. — [search result summary for arXiv 1102.0075](https://arxiv.org/pdf/1102.0075)

### Inferences
- **Codimension 1 reduces to normal agreement.** For 3D sheets in R⁴ the tangent space is fixed by a unit normal n_i. The O(3) alignment between neighbours is determined (up to an in-plane rotation) by the rotation taking n_i to n_j, so the VDM edge affinity carries the same information as |n_i·n_j|. Expect little gain over the user's "spectral-normal" or region-growing on normals from VDM *as an embedding*.
- **Frustration is the new signal.** Compute each edge's transport residual, ‖O_ij·T_j − T_i‖ (or the sheaf-Laplacian local energy), and its cycle inconsistency. Large residuals along a band of edges mark a crease even where per-point normals are noisy. These can become edge weights in a minimax-path (LAPD/LLPD) clustering instead of a summed-affinity Laplacian.
- **Creases (codimension 2).** For two-contact samples the tangent space is 2D in R⁴, and a 2D-vs-3D alignment is rank-deficient. A connection Laplacian with *variable fibre dimension* is naturally a cellular sheaf. Hansen–Ghrist sheaves allow per-vertex stalks of different dimension, so a sheaf Laplacian could encode "sheet point ↔ crease point" restriction maps. This is a research idea, not an existing tool.
- **Smooth transitions.** No tangent-bundle operator separates them, because by definition transport is consistent across them.

### Gaps
- No paper was found that uses VDM, the connection Laplacian or sheaf Laplacians to *cluster* intersecting or stratified point clouds with an empirical evaluation. Singer & Wu 2017 (spectral convergence of the connection Laplacian) was referenced only second-hand.
- No maintained Python VDM package was verified (pydiffmap implements standard diffusion maps; VDM support not checked).

---

## Q6. Riemannian / non-Euclidean multi-manifold modelling and the metric / length-scale choice on SE(3)

### Takeaway
Wang–Slavakis–Lerman (AISTATS 2015) is the main Riemannian MMC method. It builds the affinity from local sparse coding plus tangent and geodesic *directions*, with guarantees for intersecting *geodesic* submanifolds. Its efficient instances are spheres, SPD matrices and Grassmannians, not SE(3). On SE(3) there is no bi-invariant metric, and every practical metric has an explicit length scale that trades rotation against translation. No source gives a principled length-scale choice for *clustering*.

### Cited Findings
- **Riemannian multi-manifold modelling.** Wang, Slavakis, Lerman (arXiv 1410.0095; AISTATS 2015, PMLR 38) cluster data on a Riemannian manifold into groups near lower-dimensional submanifolds.
  - The affinity uses the manifold's intrinsic geometry: local sparse coding plus the directional information of tangent spaces and geodesics. Directional information is key to intersecting clusters.
  - Theory covers a simplified variant with intersecting *geodesic* submanifolds, and experiments show robustness to departures from that model.
  - The method is efficient for the sphere, SPD matrices and the Grassmannian.
  - [arXiv 1410.0095](https://www.arxiv.org/abs/1410.0095); [PMLR](https://proceedings.mlr.press/v38/wang15b.html); [Lerman MMC slides](https://www-users.cse.umn.edu/~lerman/MMC/MMC.pdf)
- **No bi-invariant metric on SE(3).** Park (1995) showed there is no bi-invariant Riemannian metric on SE(3). Chirikjian showed continuous left-invariant distances can also be invariant under right shifts by pure rotations. — [arXiv 1612.04631](https://arxiv.org/pdf/1612.04631); [Chirikjian, Partial bi-invariance of SE(3) metrics](https://rpk.lcsr.jhu.edu/wp-content/uploads/2017/08/Partial-Bi-Invariance-of-SE3-Metrics1.pdf)
- **Park–Brockett metric.** This left-invariant metric is bi-invariant on rotations and preserves R³ isotropy, but depends on a chosen length scale. — [Žefran, Kumar, Croke 1996](https://softbank.iust.ac.ir/MathBooks/Z/Zefran,%20Kumar%20and%20Croke%20-%20CHOICE%20OF%20RIEMANNIAN%20METRICS%20FOR%20RIGID%20BODY%20KINEMATICS%20-%201996.pdf)
- **Review of pose metrics.** A 2024 *Robotica* review compares and combines metrics proposed for rigid-body pose distance. — [Cambridge Robotica review](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/0EE8C14844C4842C07E54F70105013B9/S0263574723001388a.pdf/metrics-proposed-for-measuring-the-distance-between-two-rigid-body-poses-review-comparison-and-combination.pdf)
- **Pose regression loss.** The left-invariant Riemannian geodesic distance on SE(3) has been used as a loss for pose regression. — [Springer chapter](https://www.doi.org/10.1007/978-3-030-00928-1_85)

### Inferences
- **Peg (a): Riemannian machinery adds nothing.** Peg tilts are small and the reduced coordinates (rho, tilt_r, tilt_t, depth) are already a chart on SE(3)/symmetry. The user's d² = d_rho² + d_depth² + L²(d_tilt_r² + d_tilt_t²) is exactly a Park–Brockett-type metric in exponential coordinates, with L as the length scale. The real issue is L. Every tangent-angle criterion (crease angles, LAPD dihedral angles, normal agreement) is *not* invariant to L: rescaling the tilt axes changes every dihedral angle between sheets. Choose L so that the *contact-feature displacement* is isometric. One option is L ≈ the lever arm from the rotation centre to the contacting feature (tip radius for rim contacts, larger for wall contacts). Another is to test a few L values and pick the one maximising the crease angles of known creases (the analytic model can compute this without using labels for clustering).
- **Hook (b): local log charts suffice.** Use log-map coordinates at local base points (kNN graph in the left-invariant metric with an L of order the hook wire's bend radius or eyelet radius). Riemannian MMC proper is not needed, because hook modes are full-dimensional regions, not submanifolds (Q7).

### Gaps
- No source on choosing the SE(3) length scale for clustering, density estimation or manifold learning was found.
- No Riemannian MMC implementation for SE(3) was found, and code for Wang–Slavakis–Lerman was not located (Lerman's slides only).

---

## Q7. Path-based and density-robust affinities (LLPD, Fermat distances, LUND), anisotropic / local-Mahalanobis diffusion, and principled choice of K

### Takeaway
Path-based ultrametrics (LLPD) give finite-sample guarantees *including* eigengap-based selection of K and run in quasi-linear time. They separate clusters joined by *low-density* bridges, so they suit the hook's narrow passages, but by themselves they cannot separate intersecting or touching manifolds. LAPD swaps "longest leg" for "sharpest turn" to handle intersections. Fermat distances are density-weighted geodesics with spectral convergence theory and a pip package, but they reward high density, which on sampler-biased data may create artificial cuts. Anisotropic (Jacobian / local-Mahalanobis) diffusion kernels from Singer–Coifman give a principled way to make the kernel follow the local tangent space.

### Cited Findings

**LLPD (Little, Maggioni, Murphy, JMLR 21, 2020)**
- Clustering with the longest-leg path distance, which suits elongated and irregular clusters. It has finite-sample guarantees for low-dimensional clusters in high-dimensional noise with many outliers.
- Conditions are given under which the Laplacian *eigengap correctly determines the number of clusters*. The results extend to any ultrametric, and a multiscale LLPD approximation makes spectral clustering quasi-linear in n.
- [JMLR](https://jmlr.org/papers/v21/18-085.html); [arXiv 1712.06206](https://ar5iv.arxiv.org/html/1712.06206)

**LUND (Murphy & Maggioni, JMLR 20, 2019)**
- Combines diffusion distances with density/mode estimation. The diffusion time is a scale separate from the spatial scale, and there is a time window where within- and between-cluster distances separate.
- Sufficient conditions guarantee that LUND *correctly estimates K*. Inputs: density bandwidth, diffusion scale, time and threshold.
- [arXiv 1810.06702](https://arxiv.org/pdf/1810.06702); [JMLR](https://www.jmlr.org/beta/papers/v20/18-873.html)

**Fermat distances**
- García Trillos, Little, McKenzie, Murphy (JMLR 25, 2024): density-driven path metrics on Riemannian manifolds. Sample-based Fermat distances converge to continuum ones locally, at rates depending on intrinsic dimension and density weighting. Graph Laplacians built from them converge spectrally, which explains when density-driven spectral clustering works. — [arXiv 2307.05750](https://arxiv.org/abs/2307.05750); [JMLR](https://www.jmlr.org/beta/papers/v25/23-0939.html)
- Code: `pip install fermat` (F. Sapienza). Exact Floyd–Warshall is O(n³); the approximation uses Dijkstra on kNN with k = O(log n). The README states the Fermat distance "tends to follow more closely the manifold structure and regions with high density". — [PyPI fermat](https://pypi.org/project/fermat)
- Theory: Groisman, Jonckheere, Sapienza (Bernoulli 2022). Parameter choice: Chazal et al. (TMLR 2024, "Choosing the Parameter of the Fermat Distance"). — [arXiv 1810.09398](https://ar5iv.labs.arxiv.org/html/1810.09398); [PyPI page citing both](https://pypi.org/project/fermat)

**LAPD (2025)**
- The largest-*angle* path distance is the intersection-aware analogue of LLPD. It reuses the multiscale single-linkage machinery and picks K by persistence (details in Q1). — [ar5iv 2507.10710](https://ar5iv.labs.arxiv.org/html/2507.10710)

**Anisotropic / local-Mahalanobis diffusion**
- Singer & Coifman, "Non-linear independent component analysis with diffusion maps" (ACHA 2008): anisotropic diffusion kernels on the observed data approximate a Laplacian on hidden variables. Unknown distances are approximated by a symmetrised second-order (Jacobian- or covariance-corrected) form, with local Jacobians estimated by local PCA or short-burst covariances. "Many local PCAs with one global eigenmap computation." — [Singer–Coifman PDF](https://web.math.princeton.edu/~amits/publications/NLICA.pdf); [Princeton record](https://collaborate.princeton.edu/en/publications/non-linear-independent-component-analysis-with-diffusion-maps/)
- Follow-ups: Singer, Erban, Kevrekidis, Coifman (PNAS 2009) detect slow variables with anisotropic diffusion maps; Kushnir, Haddad, Coifman extend to sub-manifolds. — [Princeton record](https://collaborate.princeton.edu/en/publications/detecting-intrinsic-slow-variables-in-stochastic-dynamical-system/); [Kushnir–Haddad–Coifman PDF](https://math.ucdavis.edu/~saito/data/acha.read.s11/kushnir-haddad-coif-earth.pdf)

**Scale infrastructure**
- megaman (McQueen, Meilă, VanderPlas, Zhang, JMLR 17, 2016) does Python manifold learning with fast approximate kNN and sparse eigensolvers. It embeds millions of points in minutes on a single core (0.6M SDSS spectra in ~200 min) and includes the Coifman–Lafon density-normalised Laplacian. — [JMLR](https://www.jmlr.org/papers/v17/16-109.html); [arXiv 1603.02763](https://arxiv.org/pdf/1603.02763)

### Inferences
- **Hook-in-eyelet (b): try these first.** Modes are open 6D regions joined through narrow passages, which is the regime LLPD, LUND and Fermat-weighted spectral clustering are built for.
  - LLPD gives eigengap-based K with guarantees, provided samples are roughly uniform in C-space so that passages are genuinely sparse.
  - LUND's diffusion time handles bottlenecks that do not fully disconnect regions.
  - Both run on a kNN graph built in a left-invariant SE(3) metric (Q6).
  - The user's ToMATo-on-clearance experiment used a function (clearance), not sample density. ToMATo or LUND on *density* is the more standard variant.
- **Peg-in-hole (a): mostly the wrong signal.** Density on the peg's contact surface reflects the sampler, not the mode structure. Fermat distances, which favour high-density paths, could create cuts at sampling-density changes (false modes) and bridge across creases where density is high. Sheets are connected through creases with no density gap, so LLPD collapses them. Path-based thinking helps only in the LAPD form (angle as the leg cost), or with the tensor-voting or transport residual from Q4–Q5 as the leg cost.
- **Anisotropic kernels and mixed dimension.** A Singer–Coifman local-Mahalanobis kernel, exp(−(x_i−x_j)ᵀ(C_i⁺+C_j⁺)(x_i−x_j)/2ε) with C_i the local covariance, shrinks affinities *off* the local tangent space. Across a crease the displacement leaves each sheet's tangent space, so such kernels weaken cross-crease edges automatically. It is the soft version of Arias-Castro's projection term and also handles mixed dimension, since rank-2 covariances at crease points penalise the extra direction.
- **Principled K, best options.**
  1. Eigengap with LLPD-type guarantees.
  2. Persistence in LAPD or ToMATo.
  3. Hidalgo log-posterior for dimension layers.
  4. LUND's estimated K.
  5. For the stratum graph, the user's existing transition-graph spectral clustering, which is in effect clustering the "mode adjacency" by conductance.

### Gaps
- No source evaluates LLPD, Fermat or LUND on stratified sets with intersecting strata. LAPD is the only path method with intersection theory.
- The maintenance status of megaman and of LLPD/LUND reference code was not checked; search budget exhausted.

---

## Q8. Synthesis: which methods address the user's specific failures (mixed dimension, creases, tangent-continuous transitions, density variation), with scalability and code

### Takeaway
No single published method addresses all four failure modes. The literature supports a staged pipeline:
1. Per-point dimension or junction saliency: tensor voting eigen-gaps, or Hidalgo/GRIDE on a thinned set.
2. Within each dimension layer, an intersection-aware tangent clustering: LAPD, or Arias-Castro local-PCA spectral with K above the sheet count, or Deutsch–Medioni junction removal and relabelling.
3. Wrench/press features to raise codimension and split modes whose pose geometry is smooth.

Tangent-continuous mode boundaries have no geometric signature in pose space. Every method here will merge them, which is consistent with the user's 0.62–0.75 supervised exact-mode ceiling.

### Cited Findings
- **Mixed dimension.**
  - Hidalgo separates intersecting manifolds whose ID differs by 1 at NMI ≈ 0.84–0.89, robust to density. — [UniCatt PDF](https://publicatt.unicatt.it/retrieve/e309db6f-8864-0599-e053-3705fe0a55db/SCIREP_Hidalgo.pdf)
  - Tensor voting gives per-point dimension in N-D, robust to density and intersections. — [JMLR 2010](https://jmlr.org/papers/v11/mordohai10a.html)
  - Aamari–Berenfeld give minimax theory for mixed-dimension stratified mixtures. — [arXiv 2405.20066](https://arxiv.org/abs/2405.20066)
  - Trillos–He–Li give spectral theory for intersecting manifolds of different dimension. — [arXiv 2107.13610](https://arxiv.org/abs/2107.13610)
- **Creases (tangent jumps).**
  - LAPD separates intersecting d-manifolds with (d−1)-dimensional intersections at angle Θ, with code. — [ar5iv 2507.10710](https://ar5iv.labs.arxiv.org/html/2507.10710)
  - Arias-Castro local-PCA spectral clustering has guarantees for the threshold variants. — [ar5iv 1301.2007](https://ar5iv.labs.arxiv.org/html/1301.2007)
  - Tangent-affinity methods are reported to need angles above ~π/4. — [IJCAI 2015](https://www.ijcai.org/Proceedings/15/Papers/485.pdf)
  - The graph Laplacian degenerates near singularities. — [Belkin et al. 2012](https://proceedings.mlr.press/v23/belkin12.html)
- **Density variation.**
  - TwoNN/Hidalgo are density-insensitive. — [UniCatt PDF](https://publicatt.unicatt.it/retrieve/e309db6f-8864-0599-e053-3705fe0a55db/SCIREP_Hidalgo.pdf)
  - The Haro TPMM clusters by density *and* dimension. — [NeurIPS 2006](https://proceedings.neurips.cc/paper/2006/hash/51be2fed6c55f5aa0c16ff14c140b187-Abstract.html)
  - Fermat distances deliberately exploit density. — [PyPI fermat](https://pypi.org/project/fermat)
- **Scale (1e5–1e6).**
  - SSC-OMP/EnSC: up to 0.5M points in Python. — [GitHub ChongYou](https://github.com/ChongYou/subspace-clustering)
  - megaman: millions of points for spectral embeddings. — [JMLR](https://www.jmlr.org/papers/v17/16-109.html)
  - LLPD: quasi-linear. — [JMLR](https://jmlr.org/papers/v21/18-085.html)
  - LAPD: quasi-linear in n but exponential in d, tested to 35k points. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2507.10710)
  - Hidalgo: ~quadratic in N in practice. — [UniCatt PDF](https://publicatt.unicatt.it/retrieve/e309db6f-8864-0599-e053-3705fe0a55db/SCIREP_Hidalgo.pdf)

### Inferences

**Method-by-failure matrix** (my assessment from the cited properties; "code" = verified repo or package link in the notes above):

| Method | Linear / nonlinear | Mixed dim | Intersections | K choice | Scale (to 1e6?) | Code | Creases | Smooth transitions | Density variation |
|---|---|---|---|---|---|---|---|---|---|
| Arias-Castro local-PCA spectral | nonlinear | no (single d) | yes, angle > 0 | known K | yes, via r-packing centres | old IMA page (unverified) | yes; junction may become own cluster (good here) | no | moderate (r-ball PCA) |
| SMMC (TNN 2011) | nonlinear | no | yes | known K | kNN + MPPCA, OK | 3rd-party GitHub | yes | no | moderate |
| RMMSL (ICML 2012) | nonlinear | dim by spectral gap | yes | known K | moderate | not found | yes, "flattest" bias | partly (curvature level) | self-tuning |
| Deutsch–Medioni TV + spectral + relabel | nonlinear | yes (TV eigen-gaps) | yes, explicit junctions | known K | TV linear; spectral on smooth part | not found | yes, best fit | no | yes (TV robust) |
| LAPD (2025) | nonlinear | no (single d) | yes, (d−1)-dim | persistence | ≤ ~5e4 then propagate | GitHub HYfromLA/LAPD | yes, if Θ not small | no | denoising step |
| Trillos–He–Li annular angle-constrained graph | nonlinear | yes (theory) | yes | not addressed | kNN-graph scale | none found | yes | no | theory with density |
| SSC / EnSC / SSC-OMP | linear | yes (theory) | yes (theory) | eigengap (typical) | 1e5–5e5 | ChongYou (Python) | weak at codim 1 | no | n/a |
| SMCE | nonlinear, local | yes (neighbour count) | weak (failed in tests) | known K | moderate | MATLAB | weak | no | moderate |
| Hidalgo | ID only | yes | yes | log-posterior | ~quadratic; subsample | micheleallegra (MATLAB/Py); intRinsic (R) | only if crease has mass | no | robust |
| TPMM (Haro) | ID + density | yes | yes | not verified | EM, OK | not found | only if crease has mass | no | confounds density |
| Local homology / Euclidicity / Stolz | topological | yes | yes | n/a | heavy per-point PH | TARDIS | no for bends; yes for non-manifold points | no | partly |
| VDM / connection / sheaf Laplacian | nonlinear | no / sheaf could | partly | eigengap | megaman-scale possible | not verified | as normals + frustration | no | via α-normalisation |
| LLPD / LUND | any | n/a | no (merges touching) | eigengap / LUND K (guaranteed) | quasi-linear | not verified | no | no | robust to outliers; needs gaps |
| Fermat-distance spectral | any | n/a | no | eigengap | kNN-Dijkstra | pip `fermat` | no | no | exploits density (risky here) |
| MPPCA / MFA / K-flats / LBF | linear pieces | per-component q | yes (as flats) | BIC-style | yes | not verified | as patch borders | arbitrary splits | moderate |
| Wang–Slavakis–Lerman Riemannian MMC | geodesic submanifolds | not stated | yes | not stated | moderate | not found | yes (directions) | no | not stated |

**Suggested concrete pipeline for (a), each step tied to a cited method**
1. Thin to roughly uniform density in the pose metric (the user's "even" sampling) at 2e4–5e4 points. Fix L (Q6) by maximising known crease angles.
2. *Dimension / junction layer.*
   - Run N-D tensor voting on the 4D poses and take stick, plate and ball saliencies.
   - In parallel, run Hidalgo (q = 3, ξ = 0.8, K by log-posterior), or GRIDE with same-walk neighbours excluded.
   - Label points as sheet (3D), crease (2D) or junction (ball-salient).
3. *Within the 3D layer.* Run LAPD (d = 3, K by persistence), or Arias-Castro Algorithm 4 with the projection term, on sheet points only, with crease and junction points removed as in Deutsch–Medioni.
4. *Within the 2D layer.* Cluster crease points the same way with d = 2. Each 2D component is a candidate two-contact mode.
5. *Relabel and propagate.* Relabel removed junction points by tangent-consistency (Deutsch–Medioni ambiguity resolution), then propagate to all 1e6 samples by kNN vote.
6. *Recover smooth-boundary modes.* Add wrench / press-direction coordinates only after the geometric layer. Use them to split clusters whose pose geometry is smooth but whose contact wrench changes, then glue clusters into families with the user's existing transition-graph spectral step.

**For (b)** use a kNN graph in a left-invariant SE(3) metric with LLPD- or LUND-type clustering (eigengap or LUND-estimated K). Multi-manifold methods are not the right tool there.

**Expectation.** These steps target the two failures the user's current pipeline mixes together: mixed dimension and creases. They should raise exact-mode ARI on crease-type boundaries. They cannot recover tangent-continuous boundaries from pose alone, so wrench and contact-normal features remain essential for those.

### Gaps
- No published benchmark applies any of these methods to robotic contact-mode or C-space stratification data. Every fit statement above is an inference from method assumptions.
- The search budget ran out before I could check:
  - maintained Python code for tensor voting, MPPCA/MFA, VDM and LLPD/LUND;
  - megaman's current maintenance;
  - 2024–2026 venue-specific multi-manifold work beyond LAPD, Aamari–Berenfeld and diffusion-geometry estimators.
