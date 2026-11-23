# View the reviewer website

The publication ZIP includes the complete site at `reviewer/index.html`. The
standalone reviewer ZIP opens at `index.html`. Extract the archive before using
it. For captions and video playback, serve the extracted site with a local
static server bound to loopback. Hosted copies use HTTPS and a static server
that supports HTTP byte ranges for seeking.

A Git checkout contains the site source and omits three MP4s. To populate it,
copy these files from the publication ZIP to the same paths in the checkout:

- `reviewer/media/forge_flagship_feed.mp4`
- `reviewer/media/forge_flagship_wide.mp4`
- `reviewer/media/forge_evidence_walkthrough.mp4`

Verify the ZIP with the SHA-256 on its GitHub release. From the extracted
website directory, run `sha256sum -c SHA256SUMS.txt` to check the website files.
The export manifest also records their byte counts and SHA-256 hashes.
