# Robotics methods for identifying, segmenting and discovering contact states / formations / modes, and for learning kinematic constraints and hybrid contact models from data

Notation used below: CF = contact formation (set of principal contacts, PCs, between boundary features of two bodies); "stratum" S_A = set of configurations whose active contact set is exactly A; reciprocity = the ideal (frictionless) constraint wrench does zero power on every motion that keeps the contacts, w·V = 0. "Type" = Sup (needs mode labels), MB (model-based: needs the geometry / constraint equations per mode), Unsup (no labels, no per-mode model). Transfer notes for the user's problem are in each "Inferences" block. They are my inferences, not claims from the sources. Research note: the web-search budget ran out partway through. Items I could not verify are listed under Gaps, not presented as findings.

User problem recap (from repo docstrings, `scripts/mode_discovery_bench.py`, `knn_pose_spectral.py`, `stratum_spectral.py`, `peg_hole_analytic.py`): peg-hole samples in reduced coords (rho, radial tilt, tangential tilt, depth) with measured wrench (with friction), walk velocity and trajectory links. Every contact in the analytic model involves one of four circles (peg tip edge or shoulder edge vs hole inner or outer rim). Modes are the PC sets over those circles. Pose-only kNN spectral reaches family ARI ~0.3. Wrench or hold-direction features reach ~0.85. Hook-in-eyelet: 6-DoF hook poses near a wire loop.

## Q1. Contact-formation (CF) recognition in compliant motion: the foundations and their 2015–2026 successors (especially peg-in-hole)

### Takeaway
The classical line (KU Leuven: De Schutter, Bruyninckx, Lefebvre, Gadeyne, Meeussen; Dupont/Howe; McCarragher/Hovland; Xiao/Ji) recognises CFs model-based. A known polyhedral or curved geometry gives a CF graph and per-CF constraint equations. A Bayesian filter (Kalman/IMM, exact Bayes, particle filter, or an HMM over per-model residuals) then picks the CF whose twist–wrench reciprocity or penetration residual best explains pose and wrench data. None of these is unsupervised. Modern peg-in-hole contact-state work (2015–2026) is mostly supervised (GMM-per-phase, SVM, 1D-CNN/GRU, learned predicate classifiers) or tactile extrinsic-contact estimation. The Kyaw & Kelly (2026) stratum-graph view is the newest theory. What transfers to unsupervised discovery is the measurement model: the reciprocity residual, residual-HMMs over a transition graph, and PC lattice structure. The classifiers do not transfer.

### Cited Findings

**KU Leuven compliant-motion school (twist/wrench reciprocity + Bayesian estimation)**
- De Schutter, Bruyninckx et al., "Estimating first-order geometric parameters and monitoring contact transitions during force-controlled compliant motion," IJRR 18(12), Dec 1999.
  - Core idea: linearised Kalman filters estimate contact-normal orientations and contact-point locations, track how they vary over time, and flag transitions between contact situations.
  - Model: a general contact model covers multiple simultaneous contacts of arbitrary topology and geometry. The two main contributions are putting that model into recursive state-space form and using **the reciprocity constraint between ideal contact forces and motion freedoms as the Kalman filter measurement equation**.
  - Inputs: pose/twist plus wrench. Type: MB (contact topology known; ideal contact wrenches). Validated in full 3-D experiments.
  - Sources disagree on pages: 1161–1184 vs 1121–1184. — [SCKCEN portal](https://researchportal.sckcen.be/en/publications/estimating-first-order-geometric-parameters-and-monitoring-contac/); [Connections entry](https://connections-qj.org/node/3874)
- Mihaylova, Lefebvre, Staffetti, Bruyninckx, De Schutter (ICAR 2001; *Information & Security* 2002): an Interacting Multiple Model (IMM) estimator in which each model is one contact configuration, demonstrated on moving a cube into a corner. Type: MB. — [Lancaster ePrints](https://eprints.lancs.ac.uk/id/eprint/4347)
- Lefebvre, Bruyninckx, De Schutter:
  - "Polyhedral contact formation modeling and identification for autonomous compliant motion," IEEE TRA 19(1), 2003.
  - "Polyhedral contact formation identification for autonomous compliant motion: exact nonlinear Bayesian filtering," IEEE TRO 21(1):124–129, 2005.
  - Springer book *Nonlinear Kalman Filtering for Force-Controlled Robot Tasks* (2005, doi 10.1007/11533054).
  - Type: MB, polyhedral. — [DBLP listing via search](https://dblp.uni-trier.de/pid/90/938.html); [Springer DOI record](https://dx.doi.org/10.1007/978-3-540-77457-0_1)
- Gadeyne, Lefebvre, Bruyninckx, "Bayesian hybrid model-state estimation applied to simultaneous contact formation recognition and geometrical parameter estimation," IJRR 2005 (DOI 10.1177/0278364905056196). Hybrid discrete (CF) plus continuous (geometry) Bayesian estimation; Gadeyne's 2005 thesis covers the sequential Monte Carlo methods. Type: MB. — [Connections bib record](https://connections-qj.org/node/17752)
- Meeussen, Rutgeerts, Gadeyne, Bruyninckx, De Schutter, "Contact-state segmentation using particle filters for programming by human demonstration in compliant-motion tasks," IEEE TRO 23(2):218–231, 2007 (DOI 10.1109/TRO.2007.892227; ISER 2006 version).
  - Core idea: a particle filter jointly estimates the discrete CF and continuous geometric parameters (uncertain object poses). A contact state graph of all possible CFs supports the estimation.
  - Inputs: a 3D camera tracks the demonstration tool, and a force sensor in the tool measures the contact wrench.
  - Assumptions: polyhedral objects of known geometry.
  - Result: real-time discrimination among **245 CFs** in real experiments. Type: MB.
  - [Springer DOI page](https://dx.doi.org/10.1007/978-3-540-77457-0_1); [KU Leuven thesis](https://lirias.kuleuven.be/retrieve/67070); [thesis record](https://lirias.kuleuven.be/1746429)

**Dupont/Howe: multiple-model estimation + HMM over residuals**
- Debus, Dupont, Howe, "Contact state estimation using multiple model estimation and hidden Markov models," 8th ISER, Ischia, 2002.
  - Model: contact states are constraint equations parameterised by time-dependent sensor data and time-independent object properties. All contact-state models are fit simultaneously in a moving window by Levenberg–Marquardt nonlinear least squares on **penetration distance**. Windows whose parameter Jacobian is ill-conditioned are discarded.
  - Decision: the residuals feed an HMM "acceptance test" that combines model fit with transition probabilities from a task-based contact-state network.
  - Demo: 3D peg-in-hole. **Using only position sensing**, the contact-state sequence is recovered, along with peg dimensions and hole pose.
  - Assumptions: the geometry of the possible contact states is known; nominal parameter values are not. Type: MB.
  - The authors note the approach "provides a unified solution to the segmentation and the estimation problems." — [BU PDF](https://www.bu.edu/biorobotics/publications/02_Iser_pptyestim.pdf)

**HMM / change-detection on force signals**
- Hovland & McCarragher, "Hidden Markov models as a process monitor in robotic assembly," *Modeling, Identification and Control* 20(4):201–223, 1999.
  - Method: HMMs on dynamic force/torque signals, trained offline by Baum–Welch. A discrete event is defined as a change in contact state. The method exploits the large amount of force information at a sudden discrete-state change and accounts for noise and friction.
  - Results: 2D online experiments; reported accuracy "as high as 97[%]" with 20 training examples per event (the snippet is garbled). Type: Sup. — [MIC journal abstract](https://www.mic-journal.no/ABS/MIC-1999-4-2); [DOAJ](https://doaj.org/article/fb80376d38f54f17a280ebffc1f3040e)
- Eberman & Salisbury, "Application of change detection to dynamic contact sensing" (MIT AI Memo 1421, 1993; cited as IJRR 13(5):369–394, 1994). Strain/force signals are segmented by generalized sequential likelihood-ratio tests, and the segments are tied to contact states. The authors argue that contact forces carry substantial information about manipulation state. Type: MB/statistical, no labels for segmentation itself. — [MIT DSpace](https://dspace.mit.edu/handle/1721.1/5950); [DTIC PDF](https://apps.dtic.mil/sti/pdfs/ADA270523.pdf)

**Skubic / Volz single-ended CFs (wrist F/T only)**
- Everett, Ravuri, Volz, Skubic, "Generalized recognition of single-ended contact formations," IEEE TRA 15(5):829–836, 1999. SECF classifiers use only the wrist F/T measurement. Earlier versions needed a sizeable training set and a fixed sensor–object pose; this paper studies robustness to changes in that pose. Type: Sup. — [CAS library record](https://katalog.lib.cas.cz/KNAV/EdsRecord/edb,93052777)

**Xiao / Ji / Tang / Staffetti: automatic generation of CF graphs (the discrete structure)**
- Ji & Xiao (ICRA 1999) and Xiao & Ji (IJRR 2001):
  - Method: CF graphs are generated automatically from the two polyhedra's geometric models by a divide-and-merge approach. Goal-contact relaxation subgraphs are merged into the complete CF graph.
  - Complexity: the ICRA'99 abstract states a bound of O(N²), with N characterising object complexity.
  - Type: MB, polyhedral. — [ICRA'99 PDF](https://webpages.charlotte.edu/xiao/contactd/icra99.pdf); [IJRR PDF](https://webpages.charlotte.edu/xiao/pdf/IJRRXiaoJi.pdf)
- Principal contacts: CFs are described by PCs. "Given a valid contact formation, all of its less constrained neighboring CFs can be hypothesized topologically from the PCs." Later work extends this to planar curved objects (Tang & Xiao ICRA 2005), 3D curved objects (IJRR 2008) and point-contact state spaces between strictly curved objects (RSS 2006). — [ICRA 2004 paper](https://webpages.charlotte.edu/xiao/pdf/icra2004_w.pdf); [Tang & Xiao IJRR](https://webpages.charlotte.edu/xiao/pdf/IJRRTangXiao.pdf); [RSS 2006](https://roboticsproceedings.org/rss02/p31.html)
- Staffetti, Meeussen, Xiao, "A new formalism to characterize contact states involving articulated polyhedral objects," ICRA 2005, pp. 3619–3626 (RAS 56(5) 2008).
  - Method: oriented-matroid theory represents the **stratification of C-space into manifolds, one per contact state**. An algorithm enumerates the strata and their adjacency.
  - Caveat: a 2007 follow-up by Xiao et al. called the formalism too complex to implement. Type: MB. — [ICRA 2005 PDF](https://webpages.charlotte.edu/xiao/pdf/icra2005_e.pdf); [IROS 2007 follow-up](https://webpages.charlotte.edu/xiao/pdf/IROS07Contact.pdf)

**Kyaw & Kelly (2026): contact modes are strata**
- Phone Thiha Kyaw & Jonathan Kelly, "Contact Modes Are Strata: What Geometric Structure Buys in Discrete-Continuous Planning," arXiv 2608.15541 (v1 16 Aug 2026, v2 5 Oct 2026; spotlight at the IROS 2026 workshop on Geometric-Aware Representations).
  - Claim: "a contact mode is not merely analogous to a stratum of the configuration space; it is one."
  - Strata: S_A = {φ_i = 0 for i ∈ A, φ_i > 0 otherwise}, with dim S_A = n − |A| ("every contact that holds removes one direction"). Each sticking contact removes 2 more dimensions, giving leaves of dimension n − |A| − 2|A_s|.
  - Stratum graph: edges S_A → S_B when S_B lies in the frontier of S_A (A ⊊ B). The graph is never built: "We create an edge by motion rather than by enumeration."
  - Strata are discovered from collision-engine φ and ∇φ (MuJoCo). The planner projects onto sampled contact sets and "relabel[s] it by the constraints active there."
  - The authors note that active-set strata need not be connected.
  - Results: T-block pushing (n = 10) solved 48/48, median 760 ms; LEAP-hand cube reorientation (n = 19, 272 body pairs) solved 48/48, median 9.9 s.
  - No code link found. Type: MB (planning). — [arXiv abs](https://arxiv.org/abs/2608.15541); [arXiv HTML](https://arxiv.org/html/2608.15541)

**Modern (2014–2026) contact-state work on insertion / peg-in-hole**
- Jasim & Plapper, "Contact-state modeling of robotic assembly tasks using Gaussian mixture models," Procedia CIRP 2014.
  - Method: wrench and pose signals for the different phases of force-controlled peg-in-hole are captured; GMMs model each signal's likelihood, with EM for the parameters.
  - Setup: KUKA LWR on camshaft-cap assembly. Reported to outperform the compared modelling schemes with less computation.
  - Companion papers: an IJAMT 2014 paper argues that the signals are non-Gaussian; a 2014 paper finds the hole position via the peg-on-hole state; an IMechE-B 2015 paper extends the approach to flexible objects.
  - Type: probably Sup (one likelihood model per phase); not confirmed in full text. — [ORBilu record](https://orbilu.uni.lu/handle/10993/29967); [ORBilu 23738](https://publications.uni.lu/handle/10993/23738)
- Migimatsu, Lian, Bohg, Schaal, "Symbolic state estimation with predicates for contact-rich manipulation tasks," arXiv 2203.02468 (2022). A Bayesian estimator of symbolic task states uses learned predicate classifiers and fuses noisy multimodal observations. Real peg-in-hole and connector-socket insertion; needs little training data. Type: Sup. No code link. — [arXiv](https://arxiv.org/abs/2203.02468)
- Jin et al. (UC Berkeley), "Contact pose identification for peg-in-hole assembly under uncertainties," arXiv 2101.12467: the F/T-to-contact-pose map is **not one-to-one**, since the same force can arise from different poses. — [arXiv PDF](https://arxiv.org/pdf/2101.12467)
- Other supervised insertion work:
  - 1D-CNN + GRU on force time series for inclined-hole contact states (Biomimetic Intelligence & Robotics, 2025).
  - PolyFit (2024), supervised F/T extrinsic-pose estimation.
  - A public dual-arm peg-in-hole F/T dataset: 8 misalignment classes, 12 features.
  - SVM contact-state classification (Jakovljevic et al. 2012).
  - [BIR (DOAJ)](https://doaj.org/article/cff07c5241784c51a0d04acddba396aa); [Mendeley dataset](https://data.mendeley.com/datasets/m7g93whs3t/1)
- Kim & Rodriguez, "Active extrinsic contact sensing: application to general peg-in-hole insertion," ICRA 2022. An estimation model plus an active tactile feedback controller estimate external contacts; the controller keeps the contact in a consistent mode. An object-agnostic insertion policy is trained in simulation on sequences of contact estimates, working in contact-configuration space. Type: MB estimation + learned policy. — [arXiv 2110.03555](https://arxiv.org/abs/2110.03555v2)
- Survey: Suomalainen, Karayiannidis, Kyrki, "A survey of robot manipulation in contact," RAS 156:104224 (2022, open access). — [arXiv 2112.01942](https://arxiv.org/abs/2112.01942v3)

### Inferences
- **Reciprocity residual is the most transferable piece (De Schutter 1999 measurement equation).**
  - Convert each measured wrench to a generalized force in the reduced coordinates, τ = J(q)ᵀw, where J maps (ρ̇, tilt rates, depth rate) to the peg twist. The pairing τ·q̇ is power, so it is metric-free; only "closeness" needs a metric.
  - For an ideal contact, τ·q̇ = 0 for every motion q̇ that stays on the stratum.
  - Edge test: build kNN/walk-graph edge weights from the reciprocity residual r_ij = |τ̂_i · Δq̂_ij|. This is small when the displacement from sample i to sample j stays on i's stratum, and large when the move enters or leaves i's constraint.
  - Unlike the wrench-difference block in `knn_pose_spectral.py`, this test is directional. It fires at creases between strata even where wrench magnitude varies smoothly.
  - With Coulomb friction the residual within a mode is bounded by roughly the friction angle. For a sliding walk step, the sign and size of τ·q̇_walk estimate the dissipated friction power, which can be used to remove the friction part (as Subramani et al. do; see Q3).
- **Wrench-rank test = number of active contacts = codimension.** In a neighbourhood on the same stratum, the friction-compensated generalized forces are nonnegative combinations of the active contact normals. The numerical rank of the stacked τ's estimates |A| (1 on single-contact sheets, 2 at two-point jams). Combined with a local-PCA eigengap on poses (dimension n − |A|; see ECoMaNN in Q3), this gives two independent stratum-dimension estimates, and their agreement is a self-check.
- **HMM over walks (Debus/Meeussen pattern) as an unsupervised EM.**
  - Treat each random walk as a "demonstration". Hidden states are candidate modes; emissions are the residual of each mode's local constraint model (penetration residual, as in Debus, or reciprocity residual); transitions are restricted to a graph learned from walk links.
  - Iterate Viterbi/forward–backward labelling with refitting of the per-mode constraint models. Initialise from an over-segmentation (e.g., spectral with large k).
  - This is the Debus et al. architecture with the model bank learned instead of given.
- **Eberman–Salisbury-style GLR/SPRT change detection along each walk** on the τ sequence (or τ direction) gives change points. These can serve as cannot-link constraints across a change and must-link constraints within a segment for constrained spectral clustering. It is cheap and uses the trajectory links the user already has.
- **Xiao/Ji principal contacts.**
  - `peg_hole_analytic.py` is already a PC enumeration for curved objects (four circles × features), i.e. the Tang & Xiao curved-object case.
  - The PC lattice gives label-free structural checks for any clustering: a 2-contact cluster should (i) have one dimension less than its neighbours (eigengap / wrench rank), and (ii) be adjacent in the walk graph to the clusters of exactly its two constituent PCs.
  - This Hasse-diagram consistency can drive cluster merging and the choice of k instead of the Laplacian eigengap.
- **Kyaw & Kelly ↔ `stratum_spectral.py`.**
  - Their discovery by motion plus relabelling by active set is the planning analogue of the user's sampler.
  - The frontier relation A ⊊ B gives a directed, graded graph from higher- to lower-dimensional strata. Undirected spectral clustering of the transition graph discards this.
  - A data-side version would estimate |A| per sample (rank / eigengap above), orient edges, and require each discovered stratum to have a constant |A|.
- Supervised insertion classifiers (GMM-per-phase, CNN/GRU, SVM, predicate classifiers) are useful only as privileged-label upper bounds in the benchmark. The non-injectivity of F/T to pose (Jin et al.) is the formal reason the user's pose+wrench combination beats either alone.

### Gaps
- Not verified (search budget exhausted): Lefebvre, Bruyninckx & De Schutter, "Online statistical model recognition and state estimation for autonomous compliant motion" (believed IEEE TSMC-C 2005), including how its SPRT / likelihood-ratio tests work. Treat SPRT details as unverified.
- Skubic & Volz, "Identifying single-ended contact formations from force sensor patterns" (IEEE TRA ~2000) was not retrieved; only the related Everett et al. 1999 paper was confirmed. Hovland & McCarragher's 1997–98 ICRA HMM papers were not retrieved (only the 1999 MIC journal version).
- The journal version of Debus/Dupont/Howe (believed IJRR 2004) was not verified. The Cabras/Castellanos/Staffetti boosting CF classifier (TSMC ~2010) and a random-forest CF classifier seen in results were not opened.
- No paper was found that does **unsupervised** CF/contact-mode discovery for peg-in-hole; the one targeted search returned supervised works only.
- No public code was found for the KU Leuven CF estimators, Debus et al., Xiao CF-graph generation, or Kyaw & Kelly.

## Q2. Contact-mode enumeration and mode-guided planning: how mode boundaries are represented (hyperplane arrangements in twist space) and whether sample-based discovery can borrow the twist-cone / wrench-cone characterisation

### Takeaway
In the Mason / Huang / Cheng line, a contact mode at a configuration is a **sign vector** of the twist relative to the contact-constraint hyperplanes (+1 separating, 0 maintaining, and finer signs for sliding direction). Modes are the faces of a central hyperplane arrangement in the 6-D object-velocity space (dual to a zonotope). Sliding modes come from discretising each contact's tangent plane into sectors. Planners (CMGMP, Aceituno-Cabezas & Rodriguez MIQP, Hogan's learned mode schedules) use these local mode sets. For the sample-based problem, the transferable object is the local polyhedral cone of feasible motions and its polar, the cone of constraint wrenches. Their lineality dimension and number of extreme rays give stratum dimension and contact count directly from data.

### Cited Findings
- **Textbook definition (Lynch & Park, *Modern Robotics* §12.1.3).**
  - Each contact is labelled sliding (S), rolling (R) or breaking (B), and "the concatenation of the labels for each contact is called the contact mode."
  - Each contact splits twist space into a feasible and a penetrating half-space. For stationary fixtures, the feasible twist set is a **polyhedral convex cone**: interior twists break all contacts, twists on a face slide or roll at one contact, and twists on an edge do so at two contacts.
  - "Each contact defines a wrench and therefore a constraint plane in the three-dimensional twist space." For moving fixtures the feasible set is a general polyhedral convex set, not a cone.
  - [Modern Robotics 12.1.3](https://modernrobotics.northwestern.edu/nu-gm-book-resource/12-1-3-multiple-contacts)
- **Huang, Cheng, Mason, "Efficient contact mode enumeration in 3D," WAFR 2020** (Springer 2021). Verified from the full text:
  - Separating velocity per contact: φ̇ = [n, −n p̂] V_o. With n_c contacts these are hyperplanes through the origin in R⁶.
  - A **face of the arrangement is the set of velocities with a given sign vector in {+1, 0, −1}^k**. Contacting/separating ("cs") modes are the faces.
  - Sliding directions are approximated by dividing each tangent plane into n_t equal-angle sectors, giving 4n_t + 1 tangential modes per contact. This approximation is "essential to applications using linear approximations of the Coulomb friction model."
  - Algorithm: the arrangement is in bijection with a zonotope. Find an interior point by LP, project, convert to polar form, take the convex hull, then build the face lattice (Kaibel–Pfetsch).
  - Complexity: exponential in DoF d and polynomial in contacts n; the exponential term is shown to be unavoidable. Lattice construction dominates runtime.
  - Peg-in-hole example: a cylindrical peg *inside* a cube with a circular hole, # contact points per end, 6 DoF but effectively 2. Peg-in-hole-4 (n = 8) gives 1 cs-mode and 9 ss-modes. Box-box-2 gives 168,746 ss-modes. Hand-ball takes > 1 h.
  - No code link appears in the paper text. — [NSF PAR full text](https://par.nsf.gov/servlets/purl/10338804); [CMU RI page](https://www.ri.cmu.edu/publications/efficient-contact-mode-enumeration-in-3d)
- **Cheng, Huang, Hou, Mason, "Contact mode guided motion planning for quasidynamic dexterous manipulation in 3D," ICRA 2022 (CMGMP).**
  - Hybrid plans mix continuous transitions and discrete mode switches without pre-specified contact sequences or motion primitives.
  - Each 3D mode is split into a contacting/separating part and a sticking/sliding part. Enumeration for one object is O(N^d); typical counts are 50–400 modes per object.
  - Implementation: C++ with DART and Bullet. **Code: github.com/xianyicheng/cmgmp.**
  - A 2D precursor appeared at ICRA 2021 (arXiv 2011.01454). — [arXiv 2105.14431](https://arxiv.org/pdf/2105.14431); [2D version](https://ar5iv.labs.arxiv.org/html/2011.01454)
- **Aceituno-Cabezas & Rodriguez, "A global quasi-dynamic model for contact-trajectory optimization in manipulation," RSS 2020.**
  - Mixed-integer convex model with alternating sticking contact, simplified geometry and a quasi-dynamic relaxation (bilinear torque terms approximated). Claims global optimality, infeasibility detection, and no need for seeding.
  - A reviewer notes that object motion is given and that scaling with contacts and time steps is a concern. Type: MB. — [RSS proceedings](https://roboticsproceedings.org/rss16/p047.html); [MIT DSpace](https://dspace.mit.edu/handle/1721.1/141403)
- **Hogan, Grau, Rodriguez, "Reactive planar manipulation with convex hybrid MPC," ICRA 2018; Hogan & Rodriguez IJRR 2020 (MPC-LMS).**
  - The mixed-integer MPC is solved offline to label error states with mode schedules, and a classifier learns that map. Online, the MPC becomes a convex QP. Type: Sup (labels from an MB solver). — [arXiv 1710.05724](https://arxiv.org/pdf/1710.05724); [MIT DSpace](https://dspace.mit.edu/handle/1721.1/141404)
- **Liang, Cheng, Kroemer, "Learning preconditions of hybrid force-velocity controllers for contact-rich manipulation" (arXiv 2206.12728, 2022).**
  - HFVC skills with a relaxed synthesis framework (less precise models and pose feedback). A point-cloud precondition classifier is trained in simulation and used in a task planner.
  - Shelf-domain success: 73.2% vs 51.5% without the learned precondition. Type: Sup. — [arXiv](https://arxiv.org/abs/2206.12728)
- **Kyaw & Kelly 2026** (see Q1): sticking contacts remove two extra dimensions, so sticking vs sliding refines strata into leaves. — [arXiv HTML](https://arxiv.org/html/2608.15541)

### Inferences
- **Mode boundaries in data ≈ changes of the local feasible-twist cone.**
  - At a sample on stratum S_A, the feasible displacement set is locally a cone {q̇ : Nᵀq̇ ≥ 0}. Its **lineality space** (all-zero sign vector, "keep every contact") is the stratum's tangent space, of dimension n − |A|. Its polar is the cone of constraint wrenches, with |A| extreme rays when the normals are independent.
  - From data: (a) PCA of in-contact walk displacements and kNN offsets estimates the lineality space. (b) If the sampler records lift-off or penetration-rejected proposals, those fill in the cone interior and boundary. (c) The friction-compensated generalized forces τ = Jᵀw of neighbours are samples from the polar cone.
  - Counting extreme rays of the neighbourhood's τ-cone gives a per-sample mode-cardinality feature that does not depend on the pose metric (scale L in `knn_pose_spectral.py`). The count can use an LP membership test per vector or nonnegative factorisation.
- **Tangent-continuous boundaries are invisible to the sign-vector view.** Where two single-contact sheets meet tangentially (e.g., an edge sliding round a rim with its normal turning smoothly), the cs sign vector stays "one contact maintained" on both sides. Only the *identity* of the principal contact changes. The arrangement view therefore says these are the same local mode. Separating them needs a feature-identity cue, such as contact-point location from the wrench line of action (Q5) or a different implicit surface per PC (Q4).
- **Sliding/sticking refinement.** Friction in the measured wrench carries Huang's tangential sign information. The tangential residual should be anti-parallel to the contact velocity (walk velocity) when sliding, and anywhere in the friction cone when sticking. If the target modes are kinematic (which features touch), the friction component is a nuisance to remove. If stick/slide matters (Kyaw–Kelly leaves), it is an extra discriminating feature.
- **Enumeration as a structural check.** Given estimated normals and contact points at a sample, running the Huang algorithm (or a small LP) lists the locally feasible neighbouring modes, i.e. the star of S_A in the stratum graph. Discovered clusters should be adjacent only to clusters in that star, which is a label-free validity test for the cluster adjacency graph from `stratum_spectral.py`.
- The Huang peg-in-hole example models the peg fully inserted in the bore. The user's mouth-region modes (rim and edge contacts, jams) are a richer case that the published examples do not cover.
- For the hook, the same twist-cone reasoning applies in full 6-D. Free-space "modes" (A = ∅) are not distinguished by any cone and need connectivity analysis (Q5).

### Gaps
- No paper was found that estimates contact-mode cones or sign vectors from *sampled data* rather than from known contacts; the transfer above is my inference.
- The exact worst-case exponent in Huang et al. was garbled in text extraction (it reads like O(n^{d²/2+2.5d})). Use the abstract-level statement (exponential in d, polynomial in n).
- No public code was found for the Huang et al. enumeration; only CMGMP is public. Mason's earlier O(n²) 2D mode bound (cited inside Huang et al.) was not independently checked.

## Q3. Learning kinematic constraints and constraint manifolds from demonstrations or data (ECoMaNN, null-space learning, C-LEARN, Chou/Berenson/Ozay, Subramani et al., Sturm et al., changepoint / mixtures)

### Takeaway
Constraint-learning methods split into four families:
1. Implicit-manifold learners (ECoMaNN): local PCA tangent/normal spaces plus a level-set network.
2. Projection learners (Lin/Howard/Vijayakumar): learn the null-space projector without knowing its rank.
3. Template / model-selection learners: Sturm's articulation models with MLESAC + BIC; Subramani's pose+wrench constraint inference; C-LEARN templates; CHAMP changepoints.
4. Inverse-optimality learners (Chou/Ozay/Berenson): infer unsafe sets from optimal demonstrations.

Almost all assume a single constraint of constant dimension, or segments given by changepoints. That is exactly what fails on a stratified contact set. Their mechanisms port well per stratum or as mixtures. The pose+wrench model-selection scoring (Subramani) and the BIC merge test (Sturm) are the most directly useful.

### Cited Findings
- **ECoMaNN (Sutanto, Rayas Fernández, Englert, Ramachandran, Sukhatme, CoRL 2020, arXiv 2009.11852).**
  - Core idea: learn a level-set function h_M(q) of an equality constraint for use in a constrained sampling-based planner; learned and analytic constraints are combined via projection.
  - Training: **local PCA on kNN neighbourhoods gives tangent and normal bases**. The net is trained so that the Jacobian's null space and row space match them ("the tangent and normal spaces are equivalent to the null and row space, respectively, of the Jacobian matrix"), with projection-error losses.
  - Other losses: off-manifold augmentation along the approximate normal with a norm loss, Siamese reflection/consistency losses, and an orthogonal-subspace-alignment step to orient normals consistently.
  - **Codimension is chosen by the eigengap of the local covariance**, under the stated assumption that "the intrinsic dimensionality of the manifold at each point remains constant."
  - Scope: one network per equality constraint (sequences handled by SMP*). Needs only on-manifold data; inequality constraints are a stated limitation.
  - Code: github.com/gsutanto/smp_manifold_learning. Type: Unsup w.r.t. labels, but single-manifold.
  - [ar5iv full text](https://ar5iv.arxiv.org/html/2009.11852); [USC RESL page](https://robotics.usc.edu/resl/publications/1100/); [arXiv workshop precursor 2006.07746](https://arxiv.org/abs/2006.07746v3)
- **Lin, Howard, Vijayakumar, "Learning null space projections," ICRA 2015.**
  - Learns the null-space projection matrix of a kinematically constrained system "in the absence of any prior knowledge either on the underlying policy, the geometry, or dimensionality of the constraints."
  - Follow-ups: Lin & Howard (arXiv 1607.07611) extend this to operational space. Armesto et al. (ICRA 2017; IJRR 2018) give a closed-form solution, replacing the original non-convex optimisation over a spherical parameterisation. Howard & Vijayakumar (2007) reconstruct null-space policies under dynamic task constraints.
  - Code not verified. — [ICRA 2015 PDF](https://homepages.inf.ed.ac.uk/svijayak/publications/lin-ICRA2015.pdf); [Armesto ICRA 2017](https://homepages.inf.ed.ac.uk/svijayak/publications/arnesto-ICRA2017.pdf); [IJRR 2018](https://journals.sagepub.com/doi/pdf/10.1177/0278364918784354)
- **C-LEARN (Pérez-D'Arpino & Shah, ICRA 2017).** A multi-step task is a sequence of keyframes plus geometric constraints drawn from a knowledge base, learned from a single demonstration. It handles SE(3) volumetric and CAD constraints (e.g., parallel axes). Experiments on the Optimus dual-arm robot with transfer to a simulated Atlas. Type: Sup/template-based. Code not verified. — [project page](https://ai.stanford.edu/~cdarpino/CLEARN/index.html); [PDF](https://ai.stanford.edu/~cdarpino/CLEARN/ICRA17_DArpino_CLEARN.pdf)
- **Chou, Ozay, Berenson, "Learning parametric constraints in high dimensions from demonstrations," CoRL 2019 (PMLR 100, 2020).**
  - Method: hit-and-run sampling generates lower-cost trajectories, which are deemed unsafe. An integer program then finds a parametric unsafe set consistent with the safe and unsafe data. The parameterisation can be known or grown incrementally, with bounds on conservativeness.
  - Evaluated on a 7-DOF arm, a quadrotor and planar pushing.
  - Follow-ups: "Uncertainty-aware constraint learning…" (CoRL 2020) handles up to 30-D constraints; a journal version covers grid and parametric representations.
  - Assumes demonstrations are (locally) optimal. — [PMLR](https://proceedings.mlr.press/v100/chou20a.html); [arXiv 1910.03477](https://arxiv.org/abs/1910.03477v1); [CoRL 2020 follow-up](https://proceedings.mlr.press/v155/chou21a.html)
- **Subramani, Zinn, Gleicher, "Inferring geometric constraints in human demonstrations," CoRL 2018 (PMLR 87:223–236).**
  - Fits kinematic-constraint models (fixed point, axial rotation, prismatic, planar) and scores them on **position, force and moment** criteria. It "does not require information about the constraint type or contact geometry; it can determine both simultaneously." Demonstrated with instrumented tongs.
  - Extended version: Subramani, Hagenow, Gleicher, Zinn, "A method for constraint inference using pose and wrench measurements" (arXiv 2010.15916). From pose and wrench it identifies which constraints are present, when, and with what parameters, fitting arbitrary scleronomic models.
  - It reports that wrench data identifies constraints even when they last only briefly, and that reaction forces are estimated by **removing friction** from the measurements.
  - Model selection uses position, orientation, force and moment error metrics. No code link. Type: MB-library + model selection (no labels).
  - [PMLR](https://proceedings.mlr.press/v87/subramani18a.html); [arXiv 2010.15916](https://arxiv.org/abs/2010.15916)
- **Sturm, Stachniss, Burgard, "A probabilistic framework for learning kinematic models of articulated objects," JAIR 41 (2011), arXiv 1405.7705.** Verified from the full text:
  - Kinematic graphs with candidate edge models: rigid, prismatic, revolute, and a Gaussian-process model (dimensionality reduction + GP regression). Models are fit by **MLESAC**.
  - Selection by **BIC(M) = −2 log p(D | M, θ̂) + k log n**, lowest BIC wins. Tree structure comes from a minimum spanning tree over BIC edge costs.
  - **Merge test (Eq. 43): use a joint model if BIC(M₁₊₂ | D₁, D₂) < BIC(M₁ | D₁) + BIC(M₂ | D₂).** "Merging two models into one is beneficial if the joint model can explain the data equally well … while requiring only a single set of parameters."
  - Code: ROS `articulation` stack, BSD licence. — [arXiv 1405.7705](https://www.arxiv.org/abs/1405.7705); [JAIR](https://jair.org/index.php/jair/article/view/10718)
- **CHAMP (Niekum, Osentoski, Atkeson, Barto; "Online Bayesian changepoint detection for articulated motion models," ~ICRA 2015).** Approximate online Bayesian changepoint detection (after Fearnhead & Liu) using maximum-likelihood parameters per segment, obtained by regression or sample consensus. Articulation models are used as segment models. **Code: ROS `changepoint` package.** Type: Unsup segmentation. — [paper PDF](https://people.cs.umass.edu/~sniekum/pubs/CPD15.pdf); [ROS wiki mirror](https://whiteoak.umd.edu/roswiki/changepoint.html); [CMU TR](https://www.ri.cmu.edu/publications/champ-changepoint-detection-using-approximate-model-parameters)
- **MICAH (Jain & Niekum, arXiv 1907.09014)** learns hybrid object kinematics from **unsegmented** motion data for a hierarchical POMDP planner. **ScrewNet** (Jain, Lioutikov, Niekum, arXiv 2008.10518) estimates articulation from depth images using screw theory without a known category. — [MICAH](https://arxiv.org/pdf/1907.09014); [ScrewNet](https://arxiv.org/pdf/2008.10518)

### Inferences
- **ECoMaNN's stated assumption is exactly what fails here.** The user's set is a union of 3-D sheets and 2-D intersections, so a single ECoMaNN will smear across strata. Two direct adaptations:
  - **Use the measured wrench as the normal-space label.** The generalized force τ lies in the row space of the constraint Jacobian, so align ∇h with τ instead of the PCA normal. PCA normals mix sheets near intersections and tangent-continuous junctions; τ does not.
  - **Mixture of implicit constraints.** Use K level-set networks h_k, assign each sample to argmin_k |h_k(q)| (soft, via EM), and train each with ECoMaNN-style losses. This is a learned "mixtures of constraints" model. Intersections are where two h_k vanish together; their count is the |A| of Kyaw–Kelly.
- **ECoMaNN's eigengap codimension estimate** applied per sample (not globally) is a usable stratum-dimension feature (n − |A|). It should agree with the wrench-rank estimate (Q1/Q2).
- **Null-space projection learning is the motion dual of the user's wrench insight.**
  - Random-walk velocities are random proposals projected onto the local tangent space. A learned N(q), with rank not fixed a priori, therefore recovers the tangent projector.
  - **Jumps in N(q) (rank or subspace) mark mode boundaries without any wrench.** This is useful for the hook data, or as a second view on peg-hole.
  - The Armesto closed form makes per-neighbourhood fitting cheap.
- **Sturm's BIC merge test as the clustering engine.**
  - Over-segment (e.g., spectral with large k, or walk segments from CHAMP). Fit per patch a small family of models: (a) pose model, a low-order implicit surface in reduced coordinates; (b) wrench model, constant or affine τ direction.
  - Greedily merge walk-adjacent patches when Eq. 43 holds. This replaces eigengap model-order selection with a likelihood-based one and is naturally tied to physically meaningful per-mode models.
  - Since each peg-hole PC sheet is a low-parameter analytic surface (an edge–rim distance = 0), the correct merge boundary is where one equation stops explaining both patches. This includes some tangent-continuous junctions, where a joint fit would need extra parameters.
- **Subramani et al. is the closest published analogue of the user's "wrench is dual to tangent" approach.** Adopt its two ideas:
  - Joint scoring of a constraint hypothesis on position *and* force/moment residuals.
  - Explicit friction removal before using the wrench as a constraint indicator.
  
  Its constraint library (point, axis, prismatic, planar) would be replaced by local contact-surface models.
- **CHAMP on each random walk** turns millions of points into thousands of segments. Each segment carries a fitted constraint, so segments can be clustered (via BIC merges or spectral on segment adjacency). This uses the trajectory links more strongly than kNN pose graphs do.
- **Chou/Ozay/Berenson and C-LEARN transfer weakly.** They need optimal demonstrations or a constraint-template knowledge base. Possible niche: for the hook, Chou-style unsafe-set learning could model the free/contact boundary from samples labelled free vs in-contact, but the optimality assumption does not hold for random walks.

### Gaps
- No robotics paper explicitly titled "mixture of constraints" was verified this session (e.g., multi-constraint extensions of null-space learning or Billard-group constraint-extraction work). The mixture idea above is my inference.
- Code availability for Lin/Howard/Vijayakumar, C-LEARN, Chou et al., and Subramani et al. is unverified.
- CHAMP's venue (ICRA 2015) is inferred from the PDF filename and an earlier workshop version, not confirmed from IEEE Xplore.

## Q4. Learning hybrid contact dynamics with discrete modes (ContactNets, LCS learning, implicit learning, neural hybrid automata, residual-clustering mode discovery, phase HMMs, BP-AR-HMM)

### Takeaway
Two philosophies coexist.
1. **Implicit / complementarity learners** (ContactNets, Learning-LCS, Bianchini et al.) never cluster modes. They learn smooth per-contact quantities (signed distances, Jacobians, LCS matrices) with violation-based losses, and modes emerge as the active sets of complementarity. Code is public for the first two.
2. **Explicit mode-discovery learners** cluster states or local models and then fit per-mode dynamics and guards. Examples: Ferrari-Trecate PWA clustering, PARC, DPGMM + SVM guards (Khader et al.), GP + particle filter (Lee et al.), Neural Hybrid Automata, VHyDRO, phase-HMMs (Kroemer STAR), BP-AR-HMM (Niekum).

The explicit learners document the user's failure mode: clustering in state space over-splits modes and misses transitions without impacts. The implicit route of one learned signed-distance function per feature pair, with modes as active sets, is the most principled fit to "which feature pair is in contact" on a stratified set.

### Cited Findings
- **ContactNets (Pfrommer, Halm, Posa, CoRL 2020, PMLR 155:2279–2291).**
  - Learns **inter-body signed-distance functions and contact-frame Jacobians**, so discontinuities are encoded smoothly and implicitly.
  - Trained with a loss built on **complementarity and maximum dissipation**, not by differentiating through stiff dynamics.
  - Predicts impact, non-penetration and stiction from 60 s of real data. **Code: github.com/DAIRLab/contact-nets.** Type: Unsup (no mode labels), MB structure.
  - [arXiv 2009.11193](https://arxiv.org/pdf/2009.11193); [DAIR bib](https://dair.seas.upenn.edu/bibliography/Pfrommer2020)
- **Learning Linear Complementarity Systems (Jin, Aydinoglu, Halm, Posa, L4DC 2022, PMLR 168:1137–1149).**
  - A violation-based loss (prediction error + complementarity violation) enables gradient learning **without knowing hybrid mode boundaries**, applied to LCSs with tens of thousands of modes.
  - **Code: github.com/wanxinjin/Learning-LCS.** — [PMLR](https://proceedings.mlr.press/v168/jin22a.html); [arXiv 2112.13284](https://www.arxiv.org/abs/2112.13284?context=cs); [GitHub](https://github.com/wanxinjin/Learning-LCS)
- **Bianchini, Halm, Matni, Posa, "Generalization bounded implicit learning of nearly discontinuous functions," L4DC 2022 (PMLR 168:1112–1124).**
  - Compares one explicit and two implicit formulations. Prediction-error losses give loose generalisation bounds on stiff or near-discontinuous functions, whereas a violation-based implicit loss "can tightly bound graph distance" and has a well-conditioned landscape. — [PMLR](https://proceedings.mlr.press/v168/bianchini22a.html); [arXiv 2112.06881](https://arxiv.org/pdf/2112.06881v2)
- **Neural Hybrid Automata (Poli et al., NeurIPS 2021).**
  - Learns stochastic hybrid systems **without prior knowledge of the number of modes**, mode parameters or transition dynamics.
  - Components: a per-mode neural ODE, a discrete latent mode, and an event module, built with normalising flows and self-supervision.
  - Evaluated on mode recovery and hierarchical controllers. Code not verified. — [arXiv 2106.04165](https://arxiv.org/pdf/2106.04165); [NeurIPS](https://proceedings.neurips.cc/paper/2021/hash/5291822d0636dc429e80e953c58b6a76-Abstract.html)
- **Khader, Yin, Falco, Kragic, "Data-efficient model learning and prediction for contact-rich manipulation tasks," RA-L 2020 (arXiv 1909.04915).** Verified from the full text:
  - Mode discovery: a **Dirichlet-process GMM** (scikit-learn, K_max 10 or 20, concentration 0.1) on position and velocity of three end-effector points. A **multiclass SVM guard** predicts the next cluster from (x_t, u_t), and a multi-output GP per transition acts as the reset map.
  - Over-clustering: 8 modes on the YuMi contact task; also over-identified on a sliding mass with 3 true modes.
  - The authors state: "the identified clusters are not always uniquely corresponded to actual modes" and "not all mode transitions are accompanied by impacting contacts."
  - No code link. — [arXiv HTML v3](https://arxiv.org/html/1909.04915v3)
- **Lee, Marinho, Johnson, Gordon, Srinivasa, Mason, "Unsupervised learning for nonlinear piecewise smooth hybrid systems," arXiv 1710.00440 (2017).** GP regression per mode, including smooth manifolds across guard regions; a particle filter tracks multi-modal transitions; synthetic oversampling handles sparse transition data. Beats EKF and switching GPs. — [arXiv](https://arxiv.org/abs/1710.00440)
- **Ferrari-Trecate, Muselli, Liberati, Morari, "A clustering technique for the identification of piecewise affine and hybrid systems," Automatica 39:205–217, 2003.**
  - Combines clustering, linear identification and pattern recognition to recover both the affine submodels and the **polyhedral partition** without gridding.
  - Clustering is done in a **feature space of local model parameters**, designed so that submodels with equal coefficients on different regions can still be separated.
  - **Confidence measures on samples** down-weight unreliable local fits. — [EPFL Infoscience](https://infoscience.epfl.ch/handle/20.500.14299/132682); [UniPV](https://iris.unipv.it/handle/11571/107825)
- **PARC (Bemporad, "Piecewise linear regression and classification," IEEE TAC 2023; arXiv 2103.06189).** Multivariate piecewise-affine regression or classification over a polyhedral partition, by block-coordinate descent. **Code: `pip install pyparc`.** — [PyPI](https://pypi.org/project/pyparc/); [arXiv](https://ar5iv.labs.arxiv.org/html/2103.06189)
- **Kroemer, van Hoof, Neumann, Peters, "Learning to predict phases of manipulation tasks as hidden states," ICRA 2014 ("STAR").** Verified from the full text:
  - Model: phases are hidden states of an autoregressive HMM with phase-specific linear-Gaussian dynamics s_{t+1} ~ N(A_r s_t + B_r a_t, S_r). **Phase transitions depend on the observed state** through logistic regression, p(r_t | s_t, r_{t−1}). Learned by **EM without phase labels**.
  - Motivation: phase changes often coincide with making or breaking contact.
  - Pushing: three phases were found (free, in contact with force ramping, pushing).
  - Model order: 2 phases underfit and 4 gave redundant, flickering phases. The authors note that "incorporating a method for merging similar models would therefore be beneficial."
  - [CMU PDF](https://www.ri.cmu.edu/app/uploads/2019/03/Kroemer_ICRA_2014.pdf); [RI page](https://ri.cmu.edu/?p=112167)
- **Wang & Kroemer, "Learning robust manipulation strategies with multimodal state transition models and recovery heuristics," ICRA 2019.** A multimodal transition model is built from the task's contact dynamics and observed transitions; RL on that model plus heuristic recovery skills raised average success from 70.0% to 95.3% on three simulated tasks. — [RI PDF](https://www.ri.cmu.edu/app/uploads/2019/03/Kroemer_Wang_ICRA_2019.pdf)
- **BP-AR-HMM segmentation (Niekum, Osentoski, Konidaris, Chitta, Marthi, Barto, IJRR 34(2):131–157, 2015; IROS 2012; RSS 2013).** The Beta-Process AR-HMM **infers the number of hidden modes** and **shares modes across trajectories**. It is used to segment unstructured demonstrations into skills and to build finite-state task representations. — [IJRR PDF](https://people.cs.umass.edu/~sniekum/pubs/IJRR2014.pdf); [IROS 2012](https://people.cs.umass.edu/~sniekum/pubs/NiekumIROS2012.pdf)
- **Fazeli, Oller, Wu, Wu, Tenenbaum, Rodriguez, "See, feel, act," Science Robotics 4(26) eaav3123, 2019 (Jenga).** A temporal hierarchical Bayesian model with latent **behavioural archetypes** is learned from a short exploration phase in force and visual domains. — [MIT DSpace](https://dspace.mit.edu/handle/1721.1/126656)
- **Recent mode-recovery learners (2025–2026):**
  - **VHyDRO** (Papamichalis & Ruane, arXiv 2605.16398, May 2026): jointly infers continuous latent state and discrete contact mode with "support-safe" proposal mixing, then fits sparse port-Hamiltonian laws per regime. Segmentation is scored with **ARI, change-point F1 and segment purity** on ManiSkill and Sawyer/BridgeData. No code link.
  - **DHAL** (Liu, Teng, Liu, Zhang, Ghaffari, arXiv 2503.01842, 2025): identifies mode switches "without trajectory segmentation or event function learning," inside an RL pipeline (quadruped skateboarding).
  - [VHyDRO](https://arxiv.org/abs/2605.16398); [DHAL](https://arxiv.org/abs/2503.01842)
- **Unsupervised contact-state estimation on legged robots:** Rotella et al. use fuzzy clustering of proprioceptive data to estimate contact probability without labels (arXiv 1709.07472; ICRA 2018). — [Rotella ICRA 2018 PDF](https://nrotella.github.io/download/Rotella_ICRA2018.pdf)

### Inferences
- **A ContactNets-style "one signed distance per feature pair" is the most principled unsupervised mode model for the user's data.**
  - Learn K smooth functions φ_k(q) (small MLPs, or parametric "edge-to-circle distance" families) from samples. Training signals:
    - on-contact samples: min_k φ_k(q_i) = 0;
    - all samples: φ_k ≥ 0 (non-penetration), with free-space samples giving φ_k > 0 (useful for the hook);
    - complementarity with the measured generalized force: τ_i ∈ cone{∇φ_k(q_i) : φ_k(q_i) ≈ 0} up to the friction cone.
  - Use violation-based losses (Jin et al.; Bianchini et al.) so no mode labels or boundaries are needed.
  - The mode of a sample is its active set {k : φ_k(q_i) ≈ 0}. Single-contact sheets have one active φ_k, two-point jams two. Each φ_k is one principal contact (feature pair).
  - **This resolves tangent-continuous boundaries**: two different φ_k can have zero sets that meet tangentially, yet remain distinct smooth functions. Point-wise clustering cannot see the junction; the active-set decomposition does.
  - Choose K by BIC or validation; the analytic model suggests K ≈ 4–8 for peg-hole.
- **Khader et al.'s caveats predict the user's pose-only result (ARI ~0.3).** State-space clustering over-splits modes and misses non-impacting (smooth) transitions. Clustering must happen in a space where modes are separable: local-model parameters (Ferrari-Trecate), wrench/reciprocity, or learned implicit functions.
- **The Ferrari-Trecate recipe formalises what the user did with local-PCA normals and adds two missing pieces:**
  - Augment local-model features with the neighbourhood centroid, so equal normals on separate regions do not merge.
  - Use confidence weights, down-weighting neighbourhoods whose local fit residual is large. Those are exactly the neighbourhoods straddling strata intersections, which contaminate spectral clustering.
  - A final classifier (SVM, or PARC's polyhedral partition) then gives boundaries.
- **PARC on wrench direction vs pose** (`pip install pyparc`) gives a quick baseline: a piecewise-affine map from reduced pose to τ direction with K polyhedral regions. It is only adequate where mode boundaries are roughly planar in reduced coordinates.
- **Phase-HMMs on random walks.**
  - Kroemer's STAR with s = (pose, τ) and state-dependent transitions lets mode boundaries be functions of pose. It is unsupervised (EM), but the number of phases needs selection; Kroemer recommends merging, so pair it with Sturm's BIC merge test (Q3).
  - BP-AR-HMM removes manual model-order selection and shares modes across the many walks. However, its AR dynamics are not informative for a noise-driven random walk.
  - Replace the AR emission with a constraint-residual emission: reciprocity residual of τ against the step, or penetration of the per-mode implicit model.
- **Benchmark alignment.** VHyDRO's evaluation (ARI + change-point F1 along trajectories + segment purity) matches the user's benchmark. Adding change-point F1 on walk sequences would credit methods that place boundaries correctly even when cluster IDs fragment.
- Neural Hybrid Automata and VHyDRO are heavyweight, built for dynamic time series. The user's samples are quasi-static, so their main value is conceptual (joint discrete+continuous inference) and as metrics.

### Gaps
- Not verified this session (search budget): Jin & Posa, "Task-driven hybrid model reduction for dexterous manipulation" (TRO 2024); Bianchini et al. 2023 on simultaneous contact/continuous dynamics learning; Kroemer et al. ICRA 2015 on hierarchical multi-phase skills; the Kroemer, Niekum & Konidaris JMLR 2021 review of robot learning for manipulation.
- Code availability for Neural Hybrid Automata, VHyDRO, DHAL, Khader et al., and the BP-AR-HMM toolbox used by Niekum is unverified.
- Jasim & Plapper (Q1) may count as an EM/GMM unsupervised baseline, but their supervision setup was not confirmed from full text (the PDF fetch failed on a certificate error).

## Q5. Tactile / force-based contact localisation and particle filtering on contact manifolds: do any identify *which feature pair* is in contact?

### Takeaway
Contact-manifold estimators (Koval's manifold particle filter, Manuelli & Tedrake's contact particle filter, Sipos & Fazeli, TacGraph, STUCCO) and extrinsic-contact sensing (Ma/Dong/Rodriguez, Kim & Rodriguez, Doshi/Taylor/Rodriguez, Neural Contact Fields) all localise *where* contact is: a point or line on a known or learned surface. Several estimate the contact type (point vs line, mode) as part of the "contact configuration." None does unsupervised discovery of feature-pair classes. Their central mechanism transfers directly: explain the measured wrench by the fewest contact points, each with a friction cone. Applied to the robot's own peg/hook surface, the wrench line of action localises the peg feature, and pose maps it into the hole frame to identify the hole feature. That is an unsupervised, friction-robust feature-pair cue that separates tangent-continuous modes.

### Cited Findings
- **Manifold particle filter (Koval, Pollard, Srinivasa; IROS 2013; IJRR 2015).**
  - Contact observations confine the state to a lower-dimensional **contact manifold**, where standard particle filters starve. The MPF samples particles directly from that manifold.
  - Manifold representations compared: rejection sampling vs **trajectory rollouts**. The rollout version makes no geometric assumptions and performs similarly to the analytic one.
  - Extension (Klingensmith et al., arXiv 1604.07224) represents the manifold implicitly with a signed-distance field for 6+ D states, and converges faster than a conventional PF during persistent contact.
  - [arXiv 1604.07224](https://arxiv.org/pdf/1604.07224); [IJRR 2015 PDF](https://personalrobotics.cs.washington.edu/publications/koval2015mpf.pdf); [representations paper](https://personalrobotics.cs.washington.edu/publications//koval2013representations.pdf)
- **Contact Particle Filter (Manuelli & Tedrake, IROS 2016; Manuelli MS thesis 2018).** Localises external contacts on a rigid-body robot from **proprioception only** (external joint torques) by searching for contact points that explain the measured torque. It tracks **multiple simultaneous contacts** (simulated Atlas) and was tested on a KUKA iiwa. — [IROS PDF](https://groups.csail.mit.edu/robotics-center/public_papers/Manuelli16.pdf); [thesis](https://groups.csail.mit.edu/robotics-center/public_papers/Manuelli18.pdf)
- **Intrinsic contact sensing from force measurements:** "Contact sensing from force measurements" (MIT AI Lab memo; IJRR 1993) is the classical work on computing the contact location on a known body surface from a wrist F/T sensor. — [MIT DSpace record](https://dspace.mit.edu/handle/1721.1/5996?show=full); [IJRR'93 PDF](https://www.centropiaggio.unipi.it/sites/default/files/sensor-IJRR93.pdf)
- **Sipos & Fazeli, IROS 2022 (arXiv 2206.01245).** Two complementary particle filters, CPFGrasp for contact location and SCOPE for object pose, using only proprioception and tactile feedback. Single- and dual-arm settings. — [arXiv](https://arxiv.org/abs/2206.01245v2)
- **TacGraph (van der Merwe, Ota, Berenson, Fazeli, Jha; arXiv 2512.23856, Dec 2025; RA-L).** A factor graph with kinematic and force-consistency factors for quasi-static rigid contact. Stated limitations: needs known object and environment geometry, assumes no relative slip and a single-point contact model, and is sensitive to initialisation. A related MERL/MIT 2023 work by Kim et al. uses a factor graph fusing tactile and kinematic data for extrinsic contact estimation and control. — [arXiv 2512.23856](https://arxiv.org/pdf/2512.23856); [MERL TR2023-025](https://shadow.merl.com/publications/docs/TR2023-025.pdf)
- **Ma, Dong, Rodriguez, "Extrinsic contact sensing with relative-motion tracking from distributed tactile measurements," ICRA 2021.**
  - Contact localisation is posed as constrained estimation from **small object motions observed by tactile sensors** ("small motions of the object … indirectly encode those constraints"), subject to non-penetration and sticking friction. Validated on **fixed point and line contacts**.
  - Cites earlier Dong et al. work that used "a self-supervised data-driven approach to estimate the contact formation and contact location of a grasped object against external features."
  - [arXiv 2103.08108](https://arxiv.org/pdf/2103.08108)
- **Doshi, Taylor, Rodriguez, "Manipulation of unknown objects via contact configuration regulation," ICRA 2022.**
  - The contact configuration is the **location, geometry and mode** of contacts among object, robot and environment.
  - The object-agnostic framework **estimates wrench and motion constraints directly**, from proprioception plus wrist F/T, for planar objects in the gravity plane.
  - Follow-up (Taylor, Doshi, Rodriguez, arXiv 2310.00798) handles intermittent robot–object contact and multiple object–environment contacts with a factor-graph estimator and limited vision. — [arXiv 2203.01203](https://arxiv.org/pdf/2203.01203); [follow-up](https://arxiv.org/abs/2310.00798v1)
- **Kim & Rodriguez, ICRA 2022** (see Q1): estimates extrinsic contacts and actively keeps contact in a consistent mode to improve the estimate. — [arXiv 2110.03555](https://arxiv.org/abs/2110.03555v2)
- **Neural Contact Fields (Higuera, Dong, Boots, Mukadam, ICRA 2023, pp. 12576–12582).**
  - A neural field gives the contact probability at query points on the object surface, from end-effector pose history, tactile images and past estimates. It is described as the first method to track **arbitrary multi-modal extrinsic contacts without assuming a contact type**. Releases the YCB-Extrinsic-Contact dataset; one code implementation is listed.
  - NCF-v2 (arXiv 2309.16652) adds sim-to-real and improves insertion-policy success by +33% (mug in cupholder) and +13% (bowl in dish rack).
  - [arXiv 2210.09297](https://arxiv.org/pdf/2210.09297); [NCF-v2](https://arxiv.org/abs/2309.16652)
- **STUCCO (Zhong, Fazeli, Berenson, RA-L 7(2):3507–3514, 2022).**
  - A particle filter over **contact locations and their object associations**, with contact detection and isolation. It revises earlier ambiguous contact-to-object assignments as new data arrive.
  - Grasp success ≥ 65% vs ≤ 5% for baselines in four simulated environments, plus a real robot. **Code: `pip install stucco`**, plus a `stucco_experiments` repo. — [arXiv 2201.10434](https://arxiv.org/pdf/2201.10434); [PyPI](https://pypi.org/project/stucco/)

### Inferences
- **Wrench line-of-action as an unsupervised feature-pair cue.** This is the strongest transferable idea from this cluster.
  - For a single point contact (frictional or not, ignoring torsional friction), the measured wrench is a pure force through the contact point. The line {p : m = p × f} is computable from w alone with no environment geometry.
  - **Peg feature:** intersect that line with the robot's own peg surface (known to the robot, so not privileged) to get the contact point on the peg, which falls on the tip edge, the shoulder edge, the side or the face.
  - **Hole feature:** map that point through the pose into the hole frame. Rim and wall contacts appear as tight 1-D clusters in (radius, height) of the contact point.
  - Because this uses the force line, not the force direction, it is **robust to friction**. The direction tilts within the friction cone, but the line still passes through the contact point.
  - It **separates tangent-continuous modes**, because the contact point jumps from one feature to another (e.g., rim → wall) even when pose and normal vary smoothly.
  - **Multi-contact detector:** for two-point jams, a single-line fit leaves a residual couple (nonzero pitch). A CPF-style search for the fewest contact points on the peg surface that explain w within friction cones then returns both |A| and the feature identities.
  - Caveats: line or patch contacts (edge-on-wall or face-on-floor at zero tilt), and the user's torque normalisation (torque about the tip / 12.5 mm) must be undone to get the physical moment arm.
- **Manifold particle filter ↔ the user's sampler.**
  - The user's random-walk sampler is Koval's "trajectory rollout" representation of the contact manifold.
  - MPF's observation model makes the active sensor/contact set index which manifold component a particle may occupy. A mode-conditioned wrench likelihood p(τ | q, mode) is therefore a natural E-step for an EM over modes on the samples.
- **CPF / TacGraph / Doshi share one estimation core:** explain the wrench by contacts on a known surface, subject to friction cones, quasi-static. In the unsupervised setting, run it on the robot-side geometry only (peg/hook) and let the environment-side features emerge as clusters of contact points in the environment frame.
- **For the hook-in-eyelet:**
  - The wire loop is a torus. A contact point in loop coordinates (angle along the loop, angle around the wire) plus the hook-side contact location (tip, inner bend, shank) from the line of action gives the feature pair.
  - STUCCO-style clustering of contact points with revisable associations fits multiple simultaneous hook–wire contacts.
  - **Free-space modes are not contact modes.** They are connected components of the A = ∅ stratum, and Kyaw & Kelly note that strata need not be connected. Find them by graph connectivity of free-space walk links, for example "hook threaded through the loop vs not". Do not use geometric clustering, and do not use the wrench, which is zero there.
- **Tactile-image methods** (NCF, Ma et al., Kim & Rodriguez) need tactile sensors the user's data lacks. Their transferable idea, that small motions encode constraints, is already in the walk-velocity / null-space view (Q3).

### Gaps
- The authorship of "Contact sensing from force measurements" (believed Bicchi, Salisbury & Brock, IJRR 12(3), 1993) is not shown in the retrieved records. The IJRR'93 PDF hosted by Centro Piaggio is consistent with that, but authors were not verified this session.
- Public code for the manifold particle filter and the Contact Particle Filter was not verified. STUCCO (PyPI) and NCF (one listed implementation) are the only confirmed code releases in this cluster.
- No surveyed method does unsupervised discovery of feature-pair classes. All assume known geometry or learn from simulated labels. The line-of-action / fewest-contacts approach above is my synthesis, not a published method found in this session.
- The Dong et al. self-supervised contact-formation estimator cited by Ma et al. was not opened. Its exact paper (possibly the IROS 2019 tactile insertion work, arXiv 1909.05426, which reportedly estimates position errors from GelSlim images) is unconfirmed.

### Cross-cutting summary: catalogue and transfer ranking
All facts in this table are sourced in the sections above. The "Transfer" column is inference.

| Method (year) | Inputs | Type | Key assumptions | Code | Transfer to user's problem |
|---|---|---|---|---|---|
| De Schutter et al. KF + reciprocity (1999) | pose/twist, wrench | MB | known contact model, ideal contact wrenches | none found | **High**: reciprocity residual edge weights; wrench-rank = \|A\| |
| Meeussen et al. PF CF segmentation (2007) | 3D tool pose, F/T | MB | polyhedral, known geometry, CF graph | none found | Medium: HMM/PF over walks with learned model bank |
| Debus–Dupont–Howe MME + HMM (2002) | position only | MB | known CS geometry, unknown params | none found | **High**: residual-HMM architecture; EM-learned model bank |
| Hovland–McCarragher HMM (1999) | F/T time series | Sup | labelled events | none found | Low (needs labels) |
| Eberman–Salisbury GLR (1993/94) | force signals | stat. | signal models | none found | Medium: change points along walks → must/cannot-link |
| Xiao–Ji CF graphs / PCs (1999–2008) | geometry | MB | polyhedral → curved | none found | Medium: lattice consistency checks for clusters |
| Kyaw–Kelly strata (2026) | φ, ∇φ (collision engine) | MB planning | contact SDFs available | no link | High (theory): directed graded stratum graph; dim = n − \|A\| |
| Huang–Cheng–Mason enumeration (2020) | contacts, normals | MB | polyhedral friction approx. | none in paper | Medium: local twist/wrench-cone features; adjacency checks |
| CMGMP (2022) | model | MB planning | quasi-static/dynamic | github.com/xianyicheng/cmgmp | Low-medium (downstream use) |
| ECoMaNN (2020) | on-manifold samples | Unsup (single manifold) | constant intrinsic dim | github.com/gsutanto/smp_manifold_learning | **High** if made a mixture and normal-supervised by τ |
| Null-space projection learning (2015–18) | states, actions | Unsup | locally constant constraint | unverified | Medium-high: rank/subspace jumps of N(q) mark modes |
| Subramani et al. (2018/2020) | pose + wrench | model selection | library of constraints; friction removed | none found | **High**: joint pose/force scoring; friction removal |
| Sturm et al. articulation + BIC (2011) | poses | model selection | parametric/GP models | ROS `articulation` (BSD) | **High**: BIC merge test as clustering engine |
| CHAMP (~2015) | trajectories | Unsup segmentation | segment models | ROS `changepoint` | High: segment walks, cluster segments |
| ContactNets (2020) | trajectories | Unsup modes, MB structure | rigid contact, complementarity | github.com/DAIRLab/contact-nets | **High**: one learned φ_k per feature pair; mode = active set |
| Learning-LCS (2022) / Bianchini (2022) | trajectories | Unsup modes | LCS / near-discontinuous | github.com/wanxinjin/Learning-LCS | Medium-high: violation losses for the φ_k mixture |
| Ferrari-Trecate PWA clustering (2003) | input–output data | Unsup | PWA | none found | High: cluster local-model params + centroid + confidence |
| PARC (2023) | regression data | Unsup partition | polyhedral regions | `pip install pyparc` | Medium baseline |
| Khader et al. DPGMM+SVM (2020) | EE pos/vel | Unsup | modes separable in state space | none found | Cautionary evidence (over-splitting) |
| Kroemer STAR (2014) / BP-AR-HMM (2015) | state/action sequences | Unsup (EM / nonparametric) | linear-Gaussian per phase | BP-AR-HMM toolbox unverified | Medium: phase-HMM over walks with constraint emissions |
| NHA (2021) / VHyDRO (2026) | time series | Unsup | dynamic hybrid systems | unverified / none | Low (quasi-static data); VHyDRO metrics useful |
| Manifold PF (2013–17) | contact sensors, pose | MB | known geometry | unverified | Medium: mode-conditioned likelihood for EM |
| Contact PF (2016) | joint torques | MB | known robot surface | unverified | **High** (as line-of-action / fewest-contacts feature-pair cue) |
| STUCCO (2022) | contact detections | MB tracking | contact isolation | `pip install stucco` | Medium (hook multi-contact) |
| Doshi/Taylor contact config. regulation (2022–23) | F/T + proprioception | MB, object-agnostic | planar, quasi-static | none found | Medium: estimate wrench/motion constraints directly |
| Neural Contact Fields (2023) | tactile images + EE pose | Sup (sim) | tactile sensing | 1 implementation listed | Low (no tactile data) |
