# Configuration-space decomposition and topological classification for hook/eyelet mode discovery

Scope: methods that split C-space free space and contact space into regions or modes, and topological invariants for interlinked or threaded objects. Each entry covers the core idea, inputs, guarantees (certified or sampling-based), how it scales to 6 DoF, code availability, and how it applies to the repo's hook/eyelet mode graph and routing. The repo context comes from the docstrings of scripts/hook_mode_graph.py, mode_graph_experiments.py, hook_topology.py and build_separation_field.py:
- 6-D poses (mm, rotation vector in deg).
- A separation field on a 1 mm / 1 deg grid, refined to half spacing in a band.
- Line-of-sight graphs with spectral clustering (normalized Laplacian, eigengap k).
- Lattice continuity graphs, ToMATo basins, and a keypoint-RMS pose metric.
- Physical labels to score against: threading bitmask TIP|CROOK|SHAFT, closest part pair, and loop-plane side.

Labels used below: "Verified" means the claim was checked against the cited page in this session. "Background" means it comes from prior knowledge, is not re-fetched here, and should be confirmed before it is quoted.

---

## Q1. Convex decomposition of C-space free space (IRIS family, clique covers, GCS): how is the region count chosen, and how are narrow passages handled?

### Takeaway
The Tedrake-group pipeline works like this. A region-growing method (IRIS, IRIS-NP, IRIS-NP2, IRIS-ZO, or the certified C-IRIS) is seeded from cliques of a sampled visibility graph (Visibility Clique Cover, VCC). The resulting polytopes are then routed through with Graphs of Convex Sets (GCS). This is the closest published analogue of the repo's "visibility graph, then spectral clustering, then convex-ish modes" pipeline.

VCC does not choose the region count in advance. Regions are added until a Monte Carlo coverage threshold α is met, and greedy clique extraction stops below a minimum clique size s_min. None of the core papers treat narrow passages specially. The 2025 edge-inflation follow-up (EI-ZO) is the one tool that guarantees a polytope contains a given collision-free segment, for example a segment through a "door". Everything except C-IRIS is probabilistic or sampling-based; C-IRIS is certified through sums-of-squares (SOS) programs.

### Cited Findings

**IRIS (Deits & Tedrake, WAFR 2014): the base algorithm**
- **Core idea and guarantees (Verified):** Drake's `Iris` "grows a convex collision-free region from a sample point, given convex obstacles". Its guarantees hold only if the seed is collision-free and the domain is bounded — [Drake IRIS group docs](https://drake.mit.edu/doxygen_cxx/group__planning__iris.html).
- **Seeding:** The VCC paper notes that "traditionally, the IRIS algorithm was initialized with an uninformed ellipsoidal metric", which is what VCC replaces with clique-derived ellipsoids — [Werner et al. 2024, arXiv html](https://arxiv.org/html/2310.02875v2).

**IRIS-NP (Petersen & Tedrake, 2023)**
- Adapts IRIS to configuration space by finding separating hyperplanes with nonlinear optimization, which also lets it handle extra nonlinear constraints — [arXiv 2303.14737](https://arxiv.org/abs/2303.14737v1).
- It deliberately accepts probabilistic certification that regions are collision-free in exchange for faster generation directly in configuration-space coordinates. It scales to a dual-arm manipulator — [arXiv 2303.14737](https://arxiv.org/abs/2303.14737v1).
- Drake exposes it as `IrisNp` — [Drake docs](https://drake.mit.edu/doxygen_cxx/group__planning__iris.html).

**C-IRIS, certified regions (Amice et al., WAFR 2022; Dai, Amice, Werner, Zhang, Tedrake, IJRR)**
- The first rigorous method that generates large convex polytopes in a rational parameterization of C-space that are certified collision-free. Certification uses SOS programs — [arXiv 2302.12219](https://arxiv.org/abs/2302.12219v2); conference version [arXiv 2205.03690](https://arxiv.org/abs/2205.03690v1).
- The journal PDF is on [MIT CSAIL](https://groups.csail.mit.edu/robotics-center/public_papers/Dai23.pdf).
- Search results list Hongkai Dai, Amice, Werner, Annan Zhang and Tedrake as authors. I found no source naming Permenter as an author — [arXiv 2302.12219](https://arxiv.org/abs/2302.12219v2).
- In Drake: `CspaceFreePolytope` finds polytopes in "tangential-configuration space" and `CspaceFreeBox` finds axis-aligned boxes — [Drake docs](https://drake.mit.edu/doxygen_cxx/group__planning__iris.html).

**IRIS-NP2 and IRIS-ZO (Werner, Cohn, Jiang et al., 2024)** — [arXiv 2410.12649](https://arxiv.org/html/2410.12649v1)
- **IRIS-ZO** is gradient-free and parallel. It samples uniformly inside the current polytope, collision-checks the samples, bisects toward the ellipsoid centre to find boundary points, and adds tangent hyperplanes at the closest ones.
- **IRIS-NP2** keeps nonlinear programming but always starts it from a feasible in-collision sample, found by a greedy or a ray collision finder.
- **Guarantee (probabilistic, not certified):** the region's fraction in collision is at most ε with failure probability at most δ. The test draws M = ⌈2 log(1/δ)/(ε τ²)⌉ uniform samples and rejects if the in-collision mean exceeds (1−τ)ε (Chernoff bound). Union bounds cover the repeated tests, and τ = 0.5 in practice.
- **Benchmarks and speed:** eight systems, including 7-DoF arms, 14-DoF 2IIWAs and 15-DoF Allegro.
  - IRIS-ZO is about 15.5× faster than IRIS-NP in the "fast" setting (90% collision-free at 90% confidence) and about 14× faster in the "precise" setting (99% at 95%).
  - 2IIWAs, fast setting: 2.0 s for IRIS-ZO vs 20.5 s for IRIS-NP.
  - IRIS-ZO was slower on Allegro in the precise setting (7.5 s vs 2.2 s).
- **Narrow passages and region count:** the paper says only that in a narrow passageway "the region may be small". It reports no region counts and no dedicated narrow-passage analysis.
- **Drake:** `IrisZo` and `IrisNp2` are marked experimental; `CommonSampledIrisOptions` defaults to num_particles = 1e3 and tau = 0.5 — [Drake docs](https://drake.mit.edu/doxygen_cxx/group__planning__iris.html), [CommonSampledIrisOptions](https://drake.mit.edu/doxygen_cxx/classdrake_1_1planning_1_1_common_sampled_iris_options.html).
- **Parameterized subspaces:** `IrisParameterizationFunction` grows regions in a parameterized subspace. IrisZo needs only a double-valued version of the map; IrisNp2 also needs an AutoDiff version — [Drake docs](https://drake.mit.edu/doxygen_cxx/classdrake_1_1planning_1_1_iris_parameterization_function.html).
- **Venue:** one search summary said WAFR 2024; my recollection is ISRR 2024. Not resolved — see the [project page](https://sites.google.com/view/fastiris).

**Visibility Clique Cover, VCC (Werner, Amice, Marcucci, Rus, Tedrake, ICRA 2024)** — all Verified from [arXiv html v2](https://arxiv.org/html/2310.02875v2)
- **Sampling:** each iteration draws K samples uniformly from the part of C_free not yet covered (K = 500 for 3-DoF, 1000 for 5-DoF, 1500 for 7-DoF). Edges are sampling-based collision checks of the straight segment between two samples.
- **Clique cover:** greedy. It repeatedly solves MaxClique as an integer linear program (Gurobi, solved to global optimality) and removes the clique. It stops when the largest remaining clique is smaller than s_min (10, or 20 for 7-DoF).
- **Seeding and inflation:**
  - Each clique gets a minimum-volume enclosing ellipsoid, computed by SDP.
  - If the ellipsoid centre is in collision, it is moved to the nearest clique vertex.
  - One IRIS / IRIS-NP iteration is then run from that ellipsoid. C-IRIS can be swapped in for certified regions.
- **Stopping and region count:** the loop runs until Monte Carlo coverage (M = 5000 free samples) reaches α. The number of regions is not set in advance; it is whatever coverage α requires. α = 1 may have no finite solution.
- **Results** (region counts are the mean of 10 runs; baseline is Iterative Obstacle Seeding, IOS):

  | System | α | Regions, IOS → VCC | Runtime in s, IOS → VCC |
  |---|---|---|---|
  | Village (convex-obstacle scene; DoF not stated in fetched text) | 0.8 | 198.0 → 93.9 | 2.7e3 → 1.7e3 |
  | 3DOF Flipper | 0.9 | 10.4 → 6.7 | 2.0e2 → 4.9e1 |
  | 5DOF UR3 | 0.75 | 90.5 → 35.1 | 1.12e4 → 5.5e2 |
  | 7DOF IIWA | 0.7 | 482.6 → 46.3 | 8.9e4 → 5.7e3 |

  For the 7-DoF case this is "10 times fewer regions" and about 16× faster.
- **Narrow passages:** not discussed. The stated failure mode is that a clique's convex hull can enclose a hole (an obstacle) even with dense sampling. Adding separating-hyperplane MILP constraints fixes this but was too expensive at scale. The method is probabilistically complete under mild assumptions.
- **Drake API:** `IrisInConfigurationSpaceFromCliqueCover` with `IrisFromCliqueCoverOptions` — [header](https://drake.mit.edu/doxygen_cxx/iris__from__clique__cover_8h.html), [options](https://drake.mit.edu/doxygen_cxx/structdrake_1_1planning_1_1_iris_from_clique_cover_options.html).
  - Defaults: coverage threshold 0.7 and iteration limit 100. The `iris_options` field is a variant of IrisOptions, IrisNp2Options or IrisZoOptions.
  - Limitations — [Drake docs](https://drake.mit.edu/doxygen_cxx/group__planning__iris.html):
    - Parameterizations are not supported for IrisZo or IrisNp2 inside the clique cover.
    - Collision-checker padding is not forwarded.
    - `MaxCliqueSolverViaMip` needs a MILP solver (Gurobi or Mosek).

**EI-ZO, edge inflation on GPUs (Werner, Cheng, Stewart, Tedrake, Rus, RSS 2025)**
- Edge Inflation Zero-Order grows collision-free line segments into full-dimensional polytopes. It builds on IRIS-ZO and uses the fact that distance to a segment is convex — [RSS 2025 proceedings](https://www.roboticsproceedings.org/rss21/p045.html), [arXiv 2504.10783](https://arxiv.org/html/2504.10783v1).
- Pipeline: dynamic roadmaps find collision-free piecewise-linear paths, which are then inflated into sequences of convex sets.
- Reported results: 17.1× faster and 27.9% more reliable than a nonlinear trajectory-optimization baseline; hardware demo on a KUKA iiwa 7 — [RSS program](https://roboticsconference.org/2025/program/papers/45/).
- One search summary called this "not from the same authors" as IRIS-ZO. The RSS listing shows Werner and Tedrake on both, so that summary is wrong.
- Drake also has `SetEdgeContainmentTerminationCondition`, which configures IRIS options so the region contains the segment between two configurations — [Drake docs](https://drake.mit.edu/doxygen_cxx/group__planning__iris.html).

**Graphs of Convex Sets, GCS (Marcucci, Petersen, von Wrangel, Tedrake, Science Robotics 2023)**
- Combines Bézier-curve properties with a GCS into a mixed-integer convex program whose convex relaxation is tight. Cheap rounding usually gives globally optimal trajectories, with optimality bounds — [arXiv 2205.04422](https://arxiv.org/abs/2205.04422v1).
- Demonstrated on a quadrotor and a 14-DoF dual arm.

**GGCS, non-Euclidean GCS (Cohn, Petersen, Simchowitz, Tedrake, RSS 2023)**
- Extends GCS to graphs of geodesically-convex sets on Riemannian manifolds, so it handles continuous revolute joints and mobile bases. The zero-curvature case reduces to a mixed-integer convex program — [arXiv 2305.06341](https://arxiv.org/abs/2305.06341v1), [RSS 2023](https://roboticsproceedings.org/rss19/p057.html).

**Contact modes in GCS (Graesdal, Chia, Marcucci, Morozov, Amice, Parrilo, Tedrake, RSS 2024)**
- Each convex set encodes the quasi-static dynamics of one contact mode, relaxed by SDP, so each path through the graph is a contact sequence. On planar pushing, plans are within a small percentage of the global optimum — [arXiv 2402.10312](https://arxiv.org/abs/2402.10312v2).

### Inferences

**How VCC compares with the repo's spectral clustering**
- VCC is a hard cover by cliques of the visibility graph: every pair inside a cluster has line of sight. The repo's normalized-Laplacian spectral clustering is a relaxed min-cut partition, so two nodes in one cluster need not see each other.
- A clique therefore certifies "convex-ish" (all pairs mutually visible, up to the clique-hole caveat). A spectral cluster only certifies "weakly connected to the rest".
- Two cheap repo experiments follow:
  1. Run greedy max-clique extraction on the existing line-of-sight graph and compare the cliques with the eigengap clusters (purity, NMI and ARI against the threading labels).
  2. Use VCC's coverage-α stopping rule as an alternative to the eigengap for choosing k.

**Why mode boundaries should sit at "doors"**
- Threading changes when a hook part passes through the hole aperture. The aperture is a narrow passage bounded by the wire, which is an obstacle.
- A single convex region can contain the straight motion that threads the tip through the hole, provided the motion is collision-free. So convex regions do not automatically keep threading labels constant.
- But convex regions cannot wrap around the wire. The decomposition is therefore forced to cut near the aperture, so polytope boundaries (and GCS edges) should line up with threading transitions.
- Adding a threading-label consistency check per region (the share of samples in each polytope with the majority label) is a direct way to score this.

**Fit to the hook data (6-D, mm and rotation-vector deg)**
- The C-space is a Euclidean chart of SE(3), valid away from 180° rotations. IRIS-type methods need no rational or rotational parameterization, so IRIS-NP2 or IRIS-ZO apply directly.
- Convexity and the MVE ellipsoid depend on the mm/deg scaling. Use the repo's 0.52 mm/deg factor, or the keypoint-RMS metric locally, before any ellipsoid step.
- IRIS-ZO needs only a collision oracle. It can be reimplemented in Python against the existing convex-part separation (`hook_eyelet_model.Separation`) or the analytic sector collision, with a margin threshold (separation ≥ s₀) so regions carry clearance.
- The tight hook/eyelet clearance will give thin polytopes. Expect many small regions near contact and a few large ones in open space, which matches the repo's "open space is one easy region" observation.

**Routing over polytopes**
- With polytopes in hand, `GcsTrajectoryOptimization` (Background: Drake class, not re-verified this session) gives the mode sequence and a smooth trajectory from one convex program. This would replace the widest-path line-of-sight search.
- Clearance can be enforced by building the regions with a separation margin.
- EI-ZO / edge containment can turn the repo's existing pocket→goal line-of-sight routes into certified-by-sampling corridors of polytopes.

**C-IRIS fit**
- C-IRIS needs a kinematic tree of revolute and prismatic joints in rational (tan θ/2) form.
- The rigid hook could be modelled as 3 prismatic + 3 revolute (roll-pitch-yaw) joints, but the RPY chart differs from the rotation-vector grid, and SOS certification cost grows quickly with geometry complexity.
- For the hook this is probably overkill unless certified "no-collision" regions are needed for the tight peg-in-hole set.

### Gaps
- No paper in this line reports the number of regions needed for a two-body rigid SE(3) problem with near-zero clearance.
- No paper evaluates whether IRIS regions align with physically meaningful modes. Every metric is coverage, region count or time.
- The IRIS-ZO venue (ISRR vs WAFR 2024) is unresolved.
- Code release for EI-ZO beyond the project page (https://sites.google.com/view/GPUPolytopes) was not verified.
- A 2025–2026 journal version of VCC, or other clique-cover follow-ups beyond EI-ZO and Drake's integration, was not found in this session's searches.

---

## Q2. Visibility roadmaps, region graphs, cell decompositions, narrow-passage identification, and C-space distance fields / learned collision boundaries

### Takeaway
Classic tools each give a concrete handle on "modes = regions separated by doors":
- **Visibility-PRM guards and connectors:** a minimal set of mutually invisible representatives.
- **Approximate convex decomposition (Lien & Amato):** a concavity tolerance in place of exact convexity.
- **Narrow-passage samplers (bridge test, Gaussian, obstacle-based, medial-axis / clearance-maximizing):** detect the doors directly.
- **Topological and persistent-homology approximations of free space (Pokorny; Varava et al.):** give multiscale connectivity and loops, and Varava's slicing over orientations gives conservative connectivity guarantees in 6-D.
- **Learned or explicit C-space distance fields (Fastron, DiffCo, CDF):** give smooth clearance in configuration units. The repo's grid already stores something similar, but in workspace units.

### Cited Findings

**Visibility-PRM (Siméon, Laumond, Nissoux, Advanced Robotics 2000)**
- Keeps "guard" nodes that cannot see one another and "connector" nodes that see at least two guards, which yields a small roadmap with high visibility coverage — described in [arXiv 2303.07751](https://arxiv.org/pdf/2303.07751) and [arXiv 2401.06021](https://arxiv.org/pdf/2401.06021), which reuse the scheme.
- Background reading: [LaValle, Planning Algorithms §5.6.2 visibility roadmap](https://lavalle.pl/planning/node245.html).
- NVIDIA Isaac's navigation stack credits this paper for its visibility graph — [Isaac docs](https://docs.nvidia.com/isaac/archive/2020.1/packages/navigation/doc/algorithm.html).
- The original paper was not fetched.

**Approximate convex decomposition, ACD (Lien & Amato; SoCG 2004 polygons, SPM 2007 polyhedra)**
- Splits a model into "approximately convex" parts up to a user-set concavity tolerance by repeatedly resolving the most significant concave feature (notch). This gives far fewer parts than exact convex decomposition with comparable downstream results — [Lien ACD page](https://cs.gmu.edu/~jmlien/research/app-cd/index.html), [Parasol Lab SPM 2007](https://parasollab.web.illinois.edu/papers/Lien-acdop-2007/).
- Motion planning is listed among the applications — [Lien dissertation record](https://tdl-ir.tdl.org/handle/1969.1/ETD-TAMU-1073).

**Narrow-passage samplers in OMPL (Verified from OMPL docs and source)**
- **`BridgeTestValidStateSampler`:** if a uniform sample is invalid, draw a Gaussian-perturbed partner. If that is also invalid, keep the midpoint when it is valid. This is the bridge test of Hsu et al., ICRA 2003 (Background citation) — [source](https://ompl.kavrakilab.org/BridgeTestValidStateSampler_8cpp_source.html).
- **`GaussianValidStateSampler`:** returns the valid member of a uniform sample and a Gaussian-perturbed partner when exactly one is valid (Boor, Overmars, van der Stappen 1999 — Background) — [OMPL ValidStateSampler](https://ompl.kavrakilab.org/core/classompl_1_1base_1_1ValidStateSampler.html).
- **`ObstacleBasedValidStateSampler`:** the last valid state on the line from a valid to an invalid sample (OBPRM-style).
- **`MaximizeClearanceValidStateSampler`:** a few attempts (default 3) to raise clearance. It is the closest built-in to medial-axis sampling (MAPRM, Wilmarth, Amato, Stiller 1999 — Background) — [OMPL samplers page](https://ompl.kavrakilab.org/samplers.html), [MaximizeClearance](https://ompl.kavrakilab.org/classompl_1_1base_1_1MaximizeClearanceValidStateSampler.html).
- The base class also lists `MinimumClearanceValidStateSampler`.

**Critical states (Ichter, Schmerling, Lee, Faust, ICRA 2020)**
- Uses betweenness centrality on roadmaps to find the few "critical" states such as doorways, learns to predict criticality from local features, and connects critical states globally in a hierarchical roadmap — [arXiv 1910.03701](https://arxiv.org/pdf/1910.03701).
- The reported gain differs between versions: "up to three orders of magnitude" over uniform sampling (arXiv) vs "two orders" — [Google Research abstract](https://research.google/pubs/learned-critical-probabilistic-roadmaps-for-robotic-motion-planning/).

**Free space of rigid objects (Varava, Carvalho, Kragic, Pokorny; IJRR submission 2020, conference 2018)**
- Builds a conservative approximation of the 3-D and 6-D C-space of a rigid object by slicing over a finite sample of orientations. Each slice uses dual diagrams of unions of balls on a uniform SO(3) grid. It detects compact path components (cages) and narrow passages — [arXiv 2002.02715](https://arxiv.org/abs/2002.02715).
- 6-D runtimes of 5–21 s or 463–1558 s depending on accuracy were reported in the 2018 version — [KTH PDF](https://www.csc.kth.se/~fpokorny/static/publications/varava2018a.pdf).
- The 2-D version reports δ-completeness in experiments — [arXiv 1710.10089](https://arxiv.org/pdf/1710.10089).
- The deterministic caging precursor proves path non-existence from the connectivity of a simplicial complex that approximates collision space — [varava2017c](https://www.csc.kth.se/~fpokorny/static/publications/varava2017c.pdf).
- No code link was found.

**Proving path non-existence and infeasibility**
- **McCarthy & Bretl (UIUC 2012):** sample in-collision configurations weighted by penetration depth and build alpha shapes inside the C-obstacle. Implemented for a 3-D Euclidean C-space, with a correctness proof — [IDEALS](https://ideals.illinois.edu/items/46510).
- **Li & Dantam (RSS 2021 and later):** learn a closed manifold inside the obstacle region that separates start from goal (an infeasibility proof), alongside bidirectional RRT — [RSS 2021](https://roboticsconference.org/2021/program/papers/064/index.html).
- **SDCL (Li & Dantam, ICRA 2023):** reuses that manifold to place samples in narrow passages, with about one order of magnitude speed-up in some scenes — [SDCL PDF](https://dyalab.mines.edu/papers/li2023sdcl.pdf).

**Persistent homology of free-space samples (Pokorny, Hawasly, Ramamoorthy, RSS 2014 / IJRR)**
- Needs only collision-free samples. It builds filtrations of simplicial complexes rather than graphs and classifies trajectories using a basis of first persistent homology. Evaluated in 2-, 3-, 4- and 6-D C-spaces and on a Baxter robot — [RSS 2014](https://roboticsproceedings.org/rss10/p54.html), [IJRR PDF](https://webhomes.maths.ed.ac.uk/~v1ranick/tda/ijrr2015.pdf).
- Related: [Topological trajectory clustering with relative persistent homology](https://www.csc.kth.se/~fpokorny/static/publications/pokorny2016b.pdf) (title-level only).

**Spectral clustering as region discovery (Brunskill, Kollar, Roy, IROS 2007)**
- Topological mapping by spectral clustering and graph partitioning into submaps. This is the mobile-robot precedent for the repo's approach — [record](https://sfbtr8.spatial-cognition.de/aigaion/index.php/publications/show/993.html). The details were not fetched.

**Learned collision boundaries and distance fields**
- **Fastron (Das, Gupta, Yip, CoRL 2017):** a kernel-perceptron C-space model with active learning, used as a proxy collision checker. At least 2× faster than a polyhedral checker and at least 8× faster than a high-precision one on a 7-DoF arm — [PMLR](https://proceedings.mlr.press/v78/das17a.html), [arXiv 1709.02316](https://arxiv.org/pdf/1709.02316).
- **Forward-kinematics kernel:** up to 3× faster motion plans — [arXiv 1910.06451](https://alphaxiv.org/abs/1910.06451).
- **DiffCo:** a differentiable kernel-perceptron proxy with multi-class labels for safety-aware trajectory optimization — [arXiv 2102.07413](https://arxiv.org/pdf/2102.07413).
- **Configuration-space distance fields, CDF (Li, Chi, Razmjoo, Calinon, RSS 2024):** signed distance in joint space with direct derivatives, plus a neural (MLP) CDF. Demonstrated on a 7-axis Franka; best-paper finalist — [arXiv 2406.01137](https://arxiv.org/abs/2406.01137v1), [project page](https://sites.google.com/view/cdfmp).

**Exact cell decompositions (Background, not fetched)**
- Vertical and cylindrical algebraic decomposition and Canny's roadmap give exact connectivity, but in practice they are limited to low-dimensional or semi-algebraic problems — [LaValle ch. 6](https://lavalle.pl/planning/ch6.pdf).

### Inferences

**Separation vs C-space distance**
- The repo's separation field is a workspace signed distance (mm between bodies), sampled on a C-space grid. CDF-style methods instead store distance in C-space units to the C-obstacle boundary.
- Both can be computed on the existing grid: run a 6-D Euclidean distance transform of the free/occupied mask under the mm/deg metric (or keypoint-RMS).
- Saddles of the C-space distance transform are the narrow-passage doors. Its ridges (the medial axis) give a skeleton graph whose branches are natural mode cores.
- Running ToMATo on C-space clearance rather than on workspace separation should make the basins less sensitive to how close the hook is to the wire versus how much room it has to move.

**A grid bridge test**
- A free grid node whose neighbours at ±r along some lattice direction (or a sampled direction) are both in collision is a door node.
- This is vectorizable over the 1 mm / 1 deg grid. It gives an explicit door set whose connected components should coincide with threading-transition boundaries, which can be scored against `hook_topology.threading`.

**Visibility-PRM guard count**
- The number of guards is an independent, sample-based estimate of the number of visibility regions. It can be compared with the eigengap k and the VCC clique count to see whether the three agree on how many modes there are.

**Varava-style orientation slicing**
- At each orientation node of the grid, compute connected components of the 3-D translational free space (cheap, and exact at grid resolution). Link components of neighbouring orientation slices when they overlap.
- The result is a mode graph with conservative connectivity semantics, as in Varava et al. Narrow passages appear as small overlaps between slices.
- This fits the hook well: at a fixed orientation, threaded and unthreaded translational components should separate whenever the hook cannot pass through the aperture at that orientation.

**Persistent H1 of the free samples**
- Free space around an eyelet loop has a non-trivial 1-cycle: go through the hole and come back around the outside. Persistent H1 of the free samples (Pokorny et al.) should detect it.
- Classifying routes by H1 class separates threading routes from bypassing routes without labels. A Vietoris–Rips complex in 6-D is expensive, so use sparse, witness or alpha-on-projection complexes.

### Gaps
- I did not fetch the original Visibility-PRM, bridge-test (Hsu et al. 2003), Gaussian (Boor et al. 1999) or MAPRM (Wilmarth et al. 1999) papers. Their metadata above is Background.
- I did not confirm whether OMPL ships a guard/connector Visibility-PRM. SPARS (sparse roadmap spanners) is in OMPL per Background knowledge, not verified.
- No study was found applying CDF or Fastron to a two-rigid-body 6-DoF problem with sub-mm clearance.
- Code availability for Fastron, DiffCo, CDF and Varava et al. was not verified, beyond the CDF project page.

---

## Q3. Topological invariants for interlinking and threading (linking integral, writhe, topology coordinates, h-signatures, caging via homology, rope/hook manipulation)

### Takeaway
Three families are usable as unsupervised features of the form "threads the loop":
1. **Pairwise Gauss linking integral (GLI) / writhe-matrix features** between discretized curves (Ho & Komura topology coordinates; Ivan et al. IJRR 2013; Zhang et al.'s bin-picking entanglement map).
2. **Biot–Savart h-signatures:** a line integral of a magnetic-like field generated by obstacle skeletons, giving homotopy classes of 3-D paths (Bhattacharya, Likhachev, Kumar, RSS 2011).
3. **Homology-based loop detection plus GLI for caging and hooking** (Pokorny, Stork, Kragic, 2013), with Mitrano & Berenson's 2024 grasp-loop signature as a recent manipulation-planning variant.

For an open hook threading a closed wire loop, the GLI of each hook part with the loop centreline is a continuous, signed, pose-differentiable "threading coordinate". Its integer part tracks the repo's discrete threading bitmask.

### Cited Findings

**Topology coordinates (Ho & Komura, Computer Graphics Forum / Eurographics 2009)**
- Embed the topological relationships of body segments (tangling, wrapping) as coordinates, which greatly reduces collision-avoidance computation for close-contact motions such as dressing or carrying — [Eurographics DL](https://diglib.eg.org/items/706d8fb2-da62-4d48-973a-a945a75a0356).
- The GLI quantifies "how much the body segments are wound around each other" — [Ho thesis, Edinburgh](https://era.ed.ac.uk/items/5e9ee8f8-e1f8-4fd3-845b-d2d79cef372b/full).

**Topology-based representations for planning (Ivan, Zarubin, Toussaint, Komura, Vijayakumar, IJRR 32(9-10) 2013)**
- Writhe and interaction-mesh representations suit wrapping tasks, but a topology space alone ignores tasks defined in other spaces. The paper combines representations in optimal control as approximate inference and maps motion back to joint space — [PDF](https://homepages.inf.ed.ac.uk/svijayak/publications/ivan-IJRR2013.pdf), [SAGE DOI 10.1177/0278364913482017](https://journals.sagepub.com/doi/abs/10.1177/0278364913482017).

**Writhe matrix for entanglement between parts (Zhang, Koyama, Domae, Wan, Harada, arXiv 2021)** — [arXiv html](https://arxiv.org/html/2106.00943v2)
- Depth edges become 3-D segments. GLI is computed per segment pair with the Klenin–Langowski segment formula, giving a writhe matrix T(i,j). The paper derives writhe (Σ T / n), density (entries above the mean) and a centre (the centre of mass of T), and combines them into an entanglement map.
- Tested on C-shaped and S-shaped industrial parts: 39/60 picks vs 32/60 for a CNN and 25/60 for graspability. No numeric writhe threshold is given.
- The paper's claimed "50% improvement" on S-shaped parts does not match its own table (10/20 vs 6/20).

**3-D homotopy classes via Biot–Savart (Bhattacharya, Likhachev, Kumar, RSS 2011, best paper)**
- Obstacle skeletons are modelled as current-carrying conductors. The line integral of the resulting Biot–Savart field along a trajectory (Ampère's law) gives a homotopy-class invariant, the h-signature, which is used to constrain or explore classes in graph search — [RSS 2011 BibTeX page](https://m.roboticsproceedings.org/rss07/p02.html), [CMU RI](https://www.ri.cmu.edu/?p=109565), [AAAI 2013 overview](https://ojs.aaai.org/index.php/AAAI/article/view/8435).
- The 2-D precursor used the Cauchy integral and residue theorem.
- I could not verify a four-author Bhattacharya/Lipsky/Ghrist/Kumar "invariants for homology classes" paper in this session; see Gaps.

**Caging and hooking via homology plus GLI (KTH)**
- **Pokorny, Stork, Kragic, ICRA 2013:** from a noisy point cloud, build a simplicial complex and find graspable loops as shortest homology generators. The GLI between the hand and the loop is the evidence of a secure caging grasp — [KTH PDF (link seen in search, returned 404 on fetch)](https://www.csc.kth.se/~fpokorny/static/publications/pokorny2013a.pdf).
- **Stork, Pokorny, Kragic, Humanoids 2013, "A topology-based object representation for clasping, latching and hooking":** a loop-based representation of objects with holes plus local volume information, used to synthesize noise-tolerant caging grasps. It depends on global topological features and is complementary to force closure. Evaluated with four hands — [KTH PDF](https://www.csc.kth.se/~fpokorny/static/publications/stork2013c.pdf).
- **Companion:** "Integrated Motion and Clasp Planning with Virtual Linking" (IROS 2013) — [KTH bibtex list](https://www.csc.kth.se/~jastork/pages/bibtex.html).

**Ropes, cables and loops**
- **Grasp Loop Signature (Mitrano & Berenson, ICRA 2024):** loops formed by robot base, grippers and attach points are extracted as graph cycles and classified topologically against obstacles to guide grasp and motion search. It beat local-geometry and finite-horizon baselines in two simulated tasks and was demonstrated on a real dual-arm cable task — [arXiv 2403.01611](https://arxiv.org/pdf/2403.01611).
- **"Rope through Loop Insertion for Robotic Knotting: A Virtual Magnetic Field Formulation":** uses a loop-generated virtual magnetic field to guide the rope end through the loop. This is title-level only; the details were not fetched — [arXiv 1611.06070](https://arxiv.org/pdf/1611.06070).

### Inferences

**A continuous threading coordinate from the GLI**
- Let L be the eyelet loop's centreline (closed) and h_P the centreline of hook part P (tip, crook or shaft; open). Let Ω_L(x) be the solid angle that L subtends at x.
- Then GLI(h_P, L) = (1/4π)∫_{h_P} ∇(Ω_L)·dl. This equals ΔΩ_L/4π along the part plus the signed number of times h_P pierces a spanning disk of L (the hole).
- Here Ω_L is taken on a single-valued branch that jumps by 4π across the spanning disk; the sign follows the loop orientation.
- So round(GLI) for each part is the signed version of the repo's threading bit (when the part's ends are well away from the hole). The fractional part varies smoothly with pose: it is near ±0.5 when one end of the part lies in the hole's plane inside the aperture, the door.
- Use cases:
  - Unsupervised features for spectral clustering or ToMATo: append three signed GLIs to the pose.
  - A progress coordinate for routing.
  - A differentiable cost for trajectory optimization (as in Ivan et al.).
- It is computable cheaply with the Klenin–Langowski segment formula on the existing densely sampled centrelines from `hook_topology.py`.

**h-signature for routes**
- The h-signature of a route (for example the tip's path from pocket to goal, closed with a fixed reference path) with respect to the loop skeleton L is a homotopy-class label: 0 for routes that go around, ±1 for routes that pass through.
- This classifies alternative routes through the mode graph and can deduplicate homotopically equivalent widest paths.

**Writhe-matrix blocks as a "which part near which part" label**
- A writhe-matrix block (hook segments × eyelet segments: top wire, sides, bottom, stem) is a soft, continuous version of the repo's "closest hook part × eyelet part" label.
- Unlike nearest-point labels, it is less sensitive to tiny pose changes when two parts are almost equidistant.

**Caveat**
- For an open hook, GLI is not a homotopy invariant of the configuration; it changes continuously, and threading can be undone by sliding.
- It is invariant only after closing curves, for example a whole route or a hook closed by a reference chord.
- In the stratified sense it is a feature, not a certificate. The KTH work avoids this by using closed loops (homology generators) on the object and the hand.

### Gaps
- No paper was found that uses GLI or h-signatures to label modes of a rigid hook threading a rigid eyelet loop. The closest are the KTH clasping/hooking work (grasping objects with holes) and the rope-insertion paper.
- The Bhattacharya–Lipsky–Ghrist–Kumar "Invariants for homology classes" paper (Background: Annals of Math. and AI, about 2013) was not confirmed by search.
- The Zarubin et al. RSS 2012 "Hierarchical motion planning in topological representations" (Background) was not verified.
- Pokorny, Kragic, Kavraki & Goldberg's "winding-augmented" high-dimensional planning with 2-D topological projections (ICRA 2016, Background) was not verified this session. It is likely relevant, since it projects high-DoF paths to 2-D winding numbers.
- The Pokorny 2013 PDF returned 404, so its details come from search snippets.

---

## Q4. Contact-space structure: C-obstacle boundary patch types, contact-state graphs for assembly, and multi-modal mode adjacency

### Takeaway
Every definition of mode adjacency treats contact space as a stratified set, with lower-dimensional strata where higher ones meet:
- **Lozano-Pérez / Donald / Canny:** the contact patches of polyhedral C-obstacle boundaries are typed by feature pairs.
- **Xiao & Ji:** contact formations (CFs) form a graph whose arcs link a CF to its neighbouring relaxations or constraints. The graph is generated automatically by divide-and-merge.
- **Hauser & Latombe:** modes are adjacent where their constraint manifolds intersect, and transitions are sampled.
- **GCS contact planning:** adjacency is set intersection.
- **Huang, Cheng, Mason:** contact modes (separating, sticking or sliding per contact) are enumerated combinatorially; the count is exponential in DoF and polynomial in contacts.

### Cited Findings

**C-obstacle boundary patch types (Background)**
- For translating polygons, C-obstacle boundary pieces come from edge–vertex (EV / VE) contacts. For polyhedra, they come from face–vertex, vertex–face and edge–edge contacts (FV, VF, EE) — [LaValle ch. 4](https://lavalle.pl/planning/ch4.pdf).
- This is the classical Lozano-Pérez (1983) / Donald (1987) / Canny type A/B/C classification. Not re-verified in this session.

**Contact-state graphs (Xiao & Ji, IJRR 20(7) 2001)**
- Divide-and-merge construction of the contact state graph between two polyhedra, as a union of goal-contact-relaxation subgraphs. Each node is a CF with a representative configuration; arcs are feasible neighbouring transitions — [PDF](https://webpages.charlotte.edu/xiao/pdf/IJRRXiaoJi.pdf).
- Extended to planar curved objects (ICRA 2005) and 3-D curved objects (Tang & Xiao, IJRR 2008) — [Tang & Xiao PDF](https://webpages.charlotte.edu/xiao/pdf/IJRRTangXiao.pdf), [ICRA 2005](https://webpages.charlotte.edu/xiao/pdf/icra2005_p.pdf).
- Companion: Ji & Xiao, "Planning Motions Compliant to Complex Contact States", IJRR 2001.

**Contact mode enumeration (Huang, Cheng, Mason, WAFR 2020)**
- The first efficient 3-D contact-mode enumeration (separating, sticking or sliding per contact). Exponential in DoF, polynomial in number of contacts. Tested on peg-in-hole, boxes and a hand — [CMU RI](https://www.ri.cmu.edu/publications/efficient-contact-mode-enumeration-in-3d), [NSF PAR](https://par.nsf.gov/servlets/purl/10338804).
- Contact-mode-guided sampling planning in 2-D enumerates modes and projects random samples onto each mode's manifold — [arXiv 2011.01454](https://arxiv.org/pdf/2011.01454).

**Multi-modal planning (Hauser & Latombe, IJRR 29(7) 2010; WAFR 2008)**
- Motion splits into modes, and a mode change happens through a continuous motion satisfying both modes' constraints. Some modes have a continuum of neighbours, which makes fixed discretization intractable, hence sampling of transitions — [Hauser & Ng-Thow-Hing 2007](https://dgp.toronto.edu/public_user/victorng/publications/2007/Hauser-2007-MultiModal.pdf), [Illinois TAMP page](https://motion.cs.illinois.edu/taskandmotion/index.html).
- Page ranges for the IJRR 2011 humanoid paper differ between sources.

**Contact modes as GCS vertices (Graesdal et al., RSS 2024)**
- Each convex set encodes one contact mode, and a path through the graph is a contact sequence — [arXiv 2402.10312](https://arxiv.org/abs/2402.10312v2).

### Inferences

**The repo's labels as coarse contact formations**
- The repo's "closest hook part × eyelet part" label (tip, crook or shaft × top wire, ±y side, bottom or stem) is a coarse CF label. Xiao-style adjacency between labels would be "CF_a and CF_b share a boundary stratum": a configuration where both pairs are simultaneously within ε of contact, i.e. a two-contact CF.
- These transition configurations can be harvested from the separation field (two parts both at separation ≤ ε). Using them as edges gives a physically defined mode adjacency to compare with spectral cluster adjacency.

**Hook/eyelet geometry (wire against wire)**
- Contacts are torus-like or cylinder-on-cylinder curved surfaces, so the relevant contact-state theory is Tang & Xiao's curved-object extension. Polyhedral FV/VF/EE typing over-segments mesh-based geometry.
- The repo's convex-part separation (around 0.5 mm conservative) also introduces spurious patch boundaries at convex-part seams, which is a known risk when labelling by "closest part".

**Peg-in-hole (symmetry-reduced 4-D contact samples)**
- Huang et al.'s enumeration gives an independent ground-truth count of contact modes per contact set, against which the unsupervised clustering of stratified contact samples can be scored.

**Hauser-style transitions as mode-graph edges**
- Sample configurations in the intersection of two candidate modes and use their existence as the edge test. This replaces "nearest-neighbour line of sight crosses a cluster boundary" with a constraint-based adjacency test.

### Gaps
- I did not fetch the Lozano-Pérez (1983), Donald (1987) or Canny (1986) originals. The patch-type statements are Background via LaValle.
- Hauser & Latombe's exact definition of mode adjacency for non-expansive spaces was not read from the full text.
- No work was found that builds a contact-state graph for wire-loop / wire-hook (torus–torus) geometry specifically.

---

## Q5. Work (2015–2026) that discovers modes or regions from sampled configurations without supervision and evaluates against ground truth

### Takeaway
Evidence is sparse. Most region-discovery work in C-space is judged by coverage, region count, planning time or success rate:
- VCC and IRIS-ZO.
- Visibility PRM, critical PRM and SDCL.
- Persistent-homology trajectory classification (classes compared against known homotopy classes).
- Varava et al. (accuracy checked by progressively finer approximations).

I found no paper that clusters sampled 6-DoF rigid-body free or contact configurations and scores the clusters against physical mode labels with purity, NMI or ARI. The repo's `mode_graph_experiments.py` protocol appears ahead of the published literature on this point. The nearest unsupervised mode-discovery work clusters sensor signals (Kroemer & Su 2018) or uses spectral clustering for topological maps (Brunskill et al. 2007).

### Cited Findings
- **Kroemer & Su (ICRA 2018):** tactile signals from exploratory movements are clustered into sensory events that correspond to contact states (modes). A manipulation graph built from primitives and modes is used to verify skills and detect novel failures — [PDF](https://www.ri.cmu.edu/app/uploads/2019/03/Kroemer_Su_ICRA_2018.pdf).
- **VCC:** evaluation is by coverage α, region count and runtime, with no semantic labels — [arXiv html](https://arxiv.org/html/2310.02875v2).
- **Pokorny et al.:** trajectories are classified by persistent-homology classes in 2–6-D C-spaces from collision-free samples alone, which is unsupervised discovery of route classes — [RSS 2014](https://roboticsproceedings.org/rss10/p54.html).
- **Varava et al.:** conservative path components and narrow passages in 6-D. Accuracy is assessed by increasingly fine approximations — [arXiv 2002.02715](https://arxiv.org/abs/2002.02715).
- **Ichter et al.:** "critical" states (doorways) are found by betweenness centrality on roadmaps, an unsupervised graph statistic later learned — [arXiv 1910.03701](https://arxiv.org/pdf/1910.03701).
- **Brunskill, Kollar, Roy:** spectral clustering for topological map segmentation — [record](https://sfbtr8.spatial-cognition.de/aigaion/index.php/publications/show/993.html).

### Inferences

**Ground truth the repo can generate that the literature lacks**
- (a) Threading bitmask, (b) closest-part pair, (c) plane side. All three are already implemented.
- (d) Signed per-part GLIs (Q3), giving a continuous ground truth.
- (e) Grid connected components at fixed orientation (Varava-style), giving a conservative connectivity ground truth.
- (f) Door nodes from a grid bridge test.

**Recommended comparison set** (each scored with the existing purity/NMI/ARI and local-disagreement harness)
1. VCC cliques on the same line-of-sight graph, with k from coverage α.
2. Visibility-PRM guards, with k = number of guards.
3. ToMATo on C-space clearance (distance transform) vs on workspace separation.
4. Spectral clustering on the line-of-sight graph with GLI features added to the edge weights.
5. Orientation-slice component graphs.

**Recommended routing comparison**
- The widest path on the line-of-sight graph (current method).
- GCS over IRIS-ZO polytopes seeded by the mode clusters.
- EI-ZO corridors around the current route.
- Compare clearance, length, number of mode switches and h-signature class.

### Gaps
- No peer-reviewed 2015–2026 paper was found that evaluates unsupervised C-space region discovery against physical mode labels, for assembly or otherwise. This may reflect search coverage (about 35 tool calls) rather than true absence.
- Possibly relevant work not verified here:
  - Hybrid-system mode identification from data, such as changepoint or HMM contact-formation estimation (Lefebvre, Bruyninckx, De Schutter, TRO 2005 — Background).
  - Gaussian-mixture contact-state monitoring for peg-in-hole (Jasim & Plapper, about 2014 — Background).
  - These evaluate against labelled contact formations but use force/torque time series, not sampled configurations.
