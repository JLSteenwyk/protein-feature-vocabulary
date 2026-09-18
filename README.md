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

The data archives contain frozen evaluation inputs, selected hidden states,
SAE feature matrices, a training-candidate FASTA and split membership. The
seven SAE checkpoints are project-trained models, not ESM foundation weights.
The separate approximately 2.12 TB historical training activation store is
not included; this distribution does not establish exact SAE retraining or
foundation-training independence. Absolute paths in historical metadata are
provenance, not portable filesystem locations.

## Restore and Test

Verify the supplied SHA256SUMS before extraction. Extract all supplied tar files
into one new project directory: paths are project-relative and contain no common
enclosing folder. Do not overwrite an existing research checkout. The outer
UPLOAD_MANIFEST.json lists each member's size and SHA-256 and archive identities.

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

DOI 10.6084/m9.figshare.32059860.v1 identifies the original materials, not this
corrected code/input selection. The author will publish the new archival version.
Do not describe the new files as publicly archived until its version and file
identities have been verified.
