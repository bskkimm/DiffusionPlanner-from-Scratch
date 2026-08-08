# Notebook Visualization Design

## Goal

Make the standalone Diffusion Planner walkthrough visually explain the complete
architecture, pipeline, and tensor I/O evolution for every notebook unit.

## Scope

Update `notebooks/diffusion_planner_walkthrough.ipynb` with a hybrid learning
path:

- An annotated global architecture map covering scene inputs, structured
  encoders, context fusion, route/time conditioning, DiT denoising, and loss.
- Compact I/O cards before every utility and model unit, including
  configuration, embeddings, stochastic depth, diffusion, normalization,
  mixer blocks, encoders, fusion, route encoding, DiT, planner loss, guidance,
  and the end-to-end trace.
- Each card records input tensor shapes, key transformations, output shapes,
  and relevant masks or semantic embeddings.
- A final executed tensor ledger records actual intermediate shapes from the
  official-dimension synthetic batch.
- Tests assert that the notebook retains the visual-map, I/O-card, and ledger
  markers while continuing to compile as a standalone notebook.

## Design Boundaries

- Keep the notebook dependency-free beyond its existing PyTorch dependency.
- Use Markdown diagrams and tables rather than rendering libraries or image
  assets so the notebook remains portable and version-control-friendly.
- Do not alter model behavior, default dimensions, or the existing executable
  architecture derivation.
- The ledger is illustrative shape instrumentation, not a trained-model
  evaluation or a nuPlan preprocessing implementation.

## Verification

Run the targeted notebook test and the full `python3 -m pytest` suite. Confirm
the notebook JSON is valid, all code cells compile, and the added marker tests
protect the visual documentation contract.
