# Literature pointers

Grouped by what the thesis needs from each paper. Read the starred ones first.

## The model being extended

- **★ Cai Z, et al. Synthetic augmentation of cancer cell line multi-omic datasets
  using unsupervised deep learning. Nat Commun 15, 10390 (2024).**
  doi:10.1038/s41467-024-54771-4. MOSA: 7 views, 1,523 lines, conditional
  decoders, view dropout, contrastive tissue term. Data: figshare
  10.6084/m9.figshare.24420580 (inputs), 24562765 (imputed outputs),
  24473005 (SHAP values, pretrained model).
- **★ MOSA package.** https://github.com/QuantitativeBiology/mosa, PyPI
  `mosalib`. Read `docs/models/mosa-vae.md` and
  `docs/development/modifying-vae.md` before touching the VAE.
- PhenPred (original MOSA code). https://github.com/QuantitativeBiology/PhenPred.
  Contains the group's earlier DIP-VAE experiment (`PhenPred/vae/DIP-VAE/`,
  `CLinesLosses.dip_vae`): a first disentanglement attempt whose metrics were
  pairwise mutual information between latent dimensions and VIM/CDH1
  correlations.

## Disentangling technical from biological factors (the design to follow)

- **★ He Z, et al. Mosaic integration and knowledge transfer of single-cell
  multimodal data with MIDAS. Nat Biotechnol 42, 1594–1605 (2024).**
  doi:10.1038/s41587-023-02040-y. Latent split into biological state `c` and
  technical variable `u`; `u` is trained to predict batch (supervised), `c` is
  made batch-free with a discriminator; decoders are conditioned on `u`, which
  allows batch-corrected and counterfactual generation. This is the template
  for `supervised_nocond` / `midas` variants.
- **★ Lopez R, Regier J, Jordan MI, Yosef N. Information constraints on
  auto-encoding variational Bayes. NeurIPS 31 (2018).** arXiv:1805.08672.
  HSIC-constrained VAE: penalise HSIC between latent and nuisance variable
  instead of adversarial training; stable, no extra network. Basis of the
  `hsic_weight` and `block_hsic_weight` penalties (used here in normalised,
  CKA form).
- Louizos C, et al. The Variational Fair Autoencoder. ICLR 2016.
  arXiv:1511.00830. MMD penalty between latent distributions of different
  nuisance groups; the ancestor of the HSIC/MMD family.
- Ganin Y, et al. Domain-adversarial training of neural networks. JMLR 17
  (2016). Gradient reversal; the "flip" adversarial mode.
- Lotfollahi M, Wolf FA, Theis FJ. scGen predicts single-cell perturbation
  responses. Nat Methods 16, 715–721 (2019). Latent arithmetic for
  counterfactual prediction (vector shifts), the simplest counterfactual
  mechanism, comparable to the block-prototype replacement used here.
- Lotfollahi M, et al. Predicting cellular responses to complex perturbations
  in high-throughput screens (CPA). Mol Syst Biol 19, e11517 (2023).
  Compositional perturbation autoencoder: basal state + additive
  embeddings for covariates, adversarial classifiers to keep the basal state
  covariate-free. Directly relevant for counterfactual decoding and for
  treating EMT as a continuous covariate.
- Xu C, et al. Probabilistic harmonization and annotation of single-cell
  transcriptomics data with deep generative models (scANVI). Mol Syst Biol 17,
  e9620 (2021). Semi-supervised latent with label information; the
  classifier-in-latent idea for the tissue block.

## Unsupervised disentanglement (for the free block and as context)

- Higgins I, et al. β-VAE. ICLR 2017. Weighting the KL term; MOSA is a β-VAE.
- Kumar A, Sattigeri P, Balakrishnan A. Variational inference of disentangled
  latent concepts from unlabeled observations (DIP-VAE). ICLR 2018.
  arXiv:1711.00848. Covariance regulariser of the aggregate posterior
  (`dip_lambda_od`, `dip_lambda_d`).
- Chen RTK, et al. Isolating sources of disentanglement in VAEs (β-TCVAE).
  NeurIPS 2018. Total-correlation decomposition of the KL term.
- Kim H, Mnih A. Disentangling by factorising (FactorVAE). ICML 2018.
- **★ Locatello F, et al. Challenging common assumptions in the unsupervised
  learning of disentangled representations. ICML 2019.** Unsupervised
  disentanglement is impossible without inductive biases or supervision;
  the reason this thesis uses labelled blocks rather than β-VAE tricks.
- Locatello F, et al. Disentangling factors of variation using few labels.
  ICLR 2020. A few labels go a long way; supports light supervision.
- Eastwood C, Williams CKI. A framework for the quantitative evaluation of
  disentangled representations (DCI). ICLR 2018. Disentanglement /
  completeness / informativeness, the conceptual basis of the probe matrix.

## Evaluation of integration and batch removal

- **★ Luecken MD, et al. Benchmarking atlas-level data integration in
  single-cell genomics. Nat Methods 19, 41–50 (2022).**
  doi:10.1038/s41592-021-01336-8. Batch-removal vs bio-conservation metrics
  (kBET, iLISI/cLISI, silhouette, PCR); `source_mixing_*` and
  `tissue_silhouette_*` follow this logic in a bulk setting.
- Tran HTN, et al. A benchmark of batch-effect correction methods for
  single-cell RNA sequencing data. Genome Biol 21, 12 (2020).
- Johnson WE, Li C, Rabinovic A. Adjusting batch effects in microarray
  expression data using empirical Bayes methods (ComBat). Biostatistics 8,
  118–127 (2007). The linear reference method any VAE correction should beat.
- Kornblith S, et al. Similarity of neural network representations revisited
  (CKA). ICML 2019. The normalised HSIC used as the independence penalty.

## Interpretation of latent spaces and gene programmes

- **★ Lotfollahi M, et al. Biologically informed deep learning to query gene
  programs in single-cell atlases (expiMap). Nat Cell Biol 25, 337–350
  (2023).** doi:10.1038/s41556-022-01072-x. Latent dimensions tied to gene
  sets through a masked linear decoder; the natural next step for giving the
  free block a biological reading.
- Seninge L, et al. VEGA is an interpretable generative model for inferring
  biological network activity in single-cell transcriptomics. Nat Commun 12,
  5684 (2021). Same idea, simpler implementation.
- Svensson V, Gayoso A, Yosef N, Pachter L. Interpretable factor models of
  single-cell RNA-seq via variational autoencoders (LDVAE). Bioinformatics 36,
  3418–3421 (2020). Linear decoder for loadings.
- Byers LA, et al. An epithelial–mesenchymal transition gene signature
  predicts resistance to EGFR and PI3K inhibitors. Clin Cancer Res 19, 279–290
  (2013); Mak MP, et al. A patient-derived, pan-cancer EMT signature. Clin
  Cancer Res 22, 609–620 (2016). Sources of the EMT marker genes.
- Gonçalves E, et al. Pan-cancer proteomic map of 949 human cell lines.
  Cancer Cell 40, 835–849 (2022). Proteomics view and the published EMT score
  (`emt_proteomics` in `.obs`).

## Multi-view VAEs and missing views

- Wu M, Goodman N. Multimodal generative models for scalable weakly-supervised
  learning (MVAE, product of experts). NeurIPS 2018. MOSA's `poe` fusion.
- Shi Y, et al. Variational mixture-of-experts autoencoders for multi-modal
  deep generative models (MMVAE). NeurIPS 2019.
- Argelaguet R, et al. MOFA+. Genome Biol 21, 111 (2020). Linear factor
  baseline available through the MOSA package (`model.type: mofa`).
- Minoura K, et al. A mixture-of-experts deep generative model for integrated
  analysis of single-cell multiomics data (scMM). Cell Rep Methods 1, 100071
  (2021).

## Cell lines, organoids and tumours (the real technical factor)

- Warren A, et al. Global computational alignment of tumor and cell line
  transcriptional profiles (Celligner). Nat Commun 12, 22 (2021). Alignment of
  cell lines to tumours with contrastive PCA + MNN; the reference method for
  the model-type factor and a source of validation pairs.
- Yu K, et al. Comprehensive transcriptomic analysis of cell lines as models
  of primary tumors across 22 tumor types. Nat Commun 10, 3574 (2019).
- Peng D, et al. Evaluating the transcriptional fidelity of cancer models.
  Genome Med 13, 73 (2021).
