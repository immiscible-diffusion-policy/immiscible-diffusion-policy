See the release-root THIRD_PARTY_NOTICES.md for full provenance.

Source files preserve inline notices where present. Full upstream notices
for this backend are in licenses/. UPSTREAM.json records source revisions.

## G1 deployment extraction

`deployment/g1/` adapts the author's `g1_dp` image-DP inference and deployment scripts. `deployment/g1/SOURCE.json` (relative to release root) records source paths, source fingerprints, release fingerprints, and external controller fingerprints. The three added DP image-policy/encoder factory files were copied from the locally modified RoboTwin DP tree; they retain the Diffusion Policy/RoboTwin notices above and are covered separately by that extraction manifest rather than the original DP backend's pinned-upstream file list. The optional G1 package has its own namespace, `idp_g1`.

The G1 adapter references the experiment's `openpi_g1_control` arm/hand controller interface and binary Dex3 presets. The controller files, Unitree SDK, camera server, and URDF are not redistributed. No redistributable license for the custom controller modules was found in the inspected checkout; obtain those dependencies separately. The adapter and this release's software checks do not certify hardware compatibility.
