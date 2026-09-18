# Protein Feature Vocabulary

Code and selected inputs for "Structure tokens modulate sparse features of
protein language models". Repository:
https://github.com/JLSteenwyk/protein-feature-vocabulary

Original code is MIT licensed; see LICENSE. Data and project-trained SAE
checkpoints are shared separately. MIT does not relicense third-party data.
Sequence/annotation inputs originate from UniProt/UniRef and structures from
AlphaFold; retain their applicable terms and attribution. Frozen structures,
sequences and matrices must not silently be replaced with newer downloads.

## Scope

Historical code is under scripts/ and src/; corrected revision entry points
are under scripts/revision/. Historical scripts are not all corrected. This
public selection excludes corrected result exports, figures, manuscripts,
private correspondence and third-party model weights by author instruction.
It is not the complete internal replay package. Document-rendering and
archived-result verification commands need excluded inputs. No scientific
accuracy claim follows solely from passing the regression suite.

The compact archive contains frozen evaluation inputs and split membership.
Large hidden-state and derived SAE feature arrays are omitted for storage quota.
The original training-candidate FASTA remains available in Figshare version 1.
Some regeneration requires additional original inputs and external foundation
models; the compact selection does not directly replay every analysis. The
seven SAE checkpoints are project-trained models, not ESM foundation weights.
The separate approximately 2.12 TB historical training activation store is
not included; this distribution does not establish exact SAE retraining or
foundation-training independence. Absolute paths in historical metadata are
provenance, not portable filesystem locations.

## Restore and Test

Verify the supplied SHA256SUMS before extraction. Extract all supplied tar files
into one new project directory: paths are project-relative and contain no common
enclosing folder. Do not overwrite an existing research checkout. The outer
COMPACT_MANIFEST.json lists each member's size and SHA-256 and archive identities.
Use COMPACT_README.md for the published selection; older upload instructions
inside the frozen code archive describe a larger, unpublished local bundle.

Use Python 3.11 and the scoped dependency files under revision/reproducibility/.
For CPU tests, install PyTorch from the CPU index, then the tested lock:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0+cpu
python -m pip install -r revision/reproducibility/requirements-cpu-tests-lock.txt
python -m pytest tests/test_revision_statistics.py tests/test_revision_portability.py tests/test_revision_training_portability.py tests/test_revision_release.py -q --deselect=tests/test_revision_statistics.py::test_journal_conversion_preserves_caption_text_and_author_statements --deselect=tests/test_revision_statistics.py::test_revised_heading_highlighting_against_submitted_baseline --deselect=tests/test_revision_release.py::test_historical_intervention_verifiers_retain_archived_hashes
```

These three internal integration tests require excluded manuscripts or corrected
result fixtures. They remain in source; deselection is explicit, not a claim
that they ran in the public checkout. Model-dependent tests require external
weights and the inference environment. Historical paid API scripts are optional;
do not run them without separately authorizing charges.

## External Models and Archive Identity

Obtain ESM-2 from facebook/esm2_t33_650M_UR50D at revision
08e4846e537177426273712802403f7ba8261b6c and ESM-3 from biohub/esm3-sm-open-v1
at revision 47f0545b2b6daf26a93439a3cd610f4f7f3d5478 using the providers'
authorized routes. The ESM-3 historical namespace is EvolutionaryScale.
Structure-token generation also needs the provider's structure encoder.
Follow applicable terms. Some historical loaders use unpinned defaults; ensure
the recorded snapshots are resolved rather than assuming offline mode pins them.

The published compact release is available at
https://doi.org/10.6084/m9.figshare.32059860.v3 (five files, approximately 0.862 GB).
Its code archive matches commit d3d1884216ef74de7f30ac050378312045d78b6b;
subsequent repository documentation updates do not change that frozen snapshot.
All five public files were SHA-256 verified by download in version 2; version 3
retains the same file IDs and checksums with the corrected title. Original
materials, including the training-candidate FASTA, remain in version 1.
