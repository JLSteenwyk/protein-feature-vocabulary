# Author Upload Instructions

1. Wait for UPLOAD_MANIFEST.json to report complete=true and every archive to
   report verified_readback=true. Run `sha256sum -c SHA256SUMS` in the bundle
   directory. Do not upload a partial build or the internal joined package.
2. Upload every listed code/data/SAE tar archive, UPLOAD_MANIFEST.json and
   SHA256SUMS to Figshare. Preserve the original published version. Prefer a new
   version of the existing record if your account supports it; otherwise create
   a separately identified corrected release without deleting original files.
3. Use the revised manuscript title and link the GitHub repository. Describe
   the selection as code, frozen input data and project SAE checkpoints, not
   corrected result exports or the complete internal reproduction package.
   Review the record's license choices against the applicable data/checkpoint
   terms rather than applying MIT indiscriminately to all assets.
4. Check that no private correspondence, manuscript archive, result figures,
   third-party foundation weights or credentials are selected. Old published
   versions are historical and are not silently replaced by this selection.
5. Publish when satisfied with the record. Send the published version link or
   DOI back for independent metadata/file verification and manuscript availability
   updates. No API token is needed. The manuscript is not submitted by this step.

Archive member names, byte counts and hashes are recorded in the manifest.
Archive shards are independently readable tar files, not split byte streams;
extract each into the same new directory after checksum verification.
