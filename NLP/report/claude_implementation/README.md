# Claude implementation, parallel to Codex

This folder contains the Claude-built counterpart to the Codex-built v3 pipeline.
Both implement the same spec from `../onestream_nlp_pipeline.md` and
`../planning/paper_scope_and_narrative.md`, with intentionally different design
choices so the team can compare two complete attempts before settling on one.

## Files

- `claude_pipeline_v3.ipynb`, the eight-phase model bench
- `claude_spot_check.ipynb`, the semi-manual before-and-after Maersk LLM trace
- `_build_notebooks.py`, the builder script. Re-run to regenerate the notebooks.
- `dify_exports/`, output destination (isolated from the Codex folder)
- `figures/`, figure output destination (isolated from the Codex folder)

## Distinctive design choices vs Codex

1. **Chunking with overlap.** Heading-aware chunker with a configurable sliding-
   window overlap (default 50 tokens). Reduces the heading-only-chunk failure
   mode flagged by Kasper.
2. **Hybrid retrieval in Arch D.** BM25 plus dense Sentence-BERT, score-merged
   via reciprocal rank fusion, instead of pure dense.
3. **Chunking ablation.** Phase 1 sweeps 200, 400, and 800 token budgets and
   reports retrieval@3 sensitivity, so the chunk size is a measured choice.
4. **Stronger SBERT default.** `all-mpnet-base-v2` rather than
   `all-MiniLM-L6-v2`. Slower but better retrieval quality on the small KB.
5. **`sklearn.pipeline.Pipeline` everywhere.** Vectoriser plus classifier wired
   together so each model is one object, easier to ship to Dify.
6. **Reproducible stratified bootstrap.** One `np.random.RandomState(SEED)`
   threaded through every CI computation.

## How to run

From this folder:

```bash
python3 _build_notebooks.py     # regenerates the two .ipynb files
jupyter notebook                # open and run
```

The notebooks default to a synthetic smoke test (`/tmp/claude_synth_*`), so they
work end-to-end with no Maersk data. Linus flips `SMOKE = False` in the config
cell and edits the path placeholders before running against real data.

## Outputs are isolated

Every figure and CSV emitted by these notebooks lands in
`claude_implementation/dify_exports/` or `claude_implementation/figures/`.
Codex's outputs land in the parent `dify_exports/` and `assets/figures/`. There
is no path collision.
