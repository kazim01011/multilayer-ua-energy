# Overleaf Manuscript Workspace

This folder contains the IEEE two-column manuscript source, bibliography, final
paper figures, and the compiled draft PDF.

Compile locally with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The paper figures used in the Results section are stored in
`overleaf/figures/results/`. MATLAB-editable versions and their source CSV
files are stored under `../matlab_figures/results/`.
