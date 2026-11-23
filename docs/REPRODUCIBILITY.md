# Reproduction and scope

## Tier A · included

Use Python 3.12, uv, Matplotlib from the supplied lockfile, and a TeX installation with `latexmk`, `pdflatex`, Poppler's `pdftotext` and `pdfinfo`. From the root:

```sh
uv sync --frozen
uv run --frozen python scripts/reproduce.py --out ../forge-reproduction
```

Choose a new output directory outside the source tree. The command verifies the archived data inventory, runs analysis tests, tallies checker and repair outcomes, checks the archived trial witnesses and fixture sensitivity, regenerates study figures, validates the included claim-source subset, and rebuilds the primary paper tables and figures. Both TeX sources are compiled and their extracted text and page counts are compared with the [research paper](../papers/current/FORGE_Research_Paper.pdf) and [technical supplement](../papers/current/FORGE_Technical_Supplement.pdf). A rebuild can have different PDF bytes because of compiler metadata.

The [claim registry](../evidence/04_CLAIM_EVIDENCE_REGISTRY.jsonl) covers the claims used to generate the paper's tables. The [evidence map](../papers/current/PAPER_CLAIM_EVIDENCE_MAP.csv) ties additional derived quantities to their frozen source keys. The data distinguishes development, qualification and held-out observations. The simulated contact model, feature-local geometry truth, whole-part truth, and finite production model have distinct scopes.

## Tier B · selected fixture pair

The `tierb/` directory includes an exact replay recipe, two data-only scene descriptions, pinned input hashes and reference records for held-out disturbance 630103. The original fixture exceeds the yaw limit and the revised lead-in fixture passes under the archived reference conditions. This was the smallest held-out seed with that outcome pattern under the predeclared selection rule.

Tier B requires 38 checksum-matched STL files. They are excluded from the distribution because redistribution rights have not been confirmed. With separately authorized copies of those files, run from the package root:

```sh
cd tierb
uv sync --frozen
uv run --frozen python replay.py run --package package --meshes /path/to/authorized/meshes --manifest inputs_manifest.json --result /tmp/forge-tierb-result.json
```

The optional replay checks the one pair with pinned MuJoCo and NumPy. It does not reproduce the population of held-out pairs, the development and qualification cohorts, or the fast-approach profile. The source's test suite checks the manifest without mesh files and runs the physical replay tests only with explicitly supplied matching inputs.

## Tier C · outside the release

The proprietary design generation, project orchestration and full simulator pipelines are excluded. No provider replay, full cohort physics regeneration, manufacturing validation, or hardware tests are part of this snapshot.
