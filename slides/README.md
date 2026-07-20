# Slides

The tutorial slide deck is a single Beamer file:

```bash
slides/main.tex
```

Install Tectonic and Pygments with conda:

```bash
conda install -c conda-forge tectonic pygments
```

or, if using mamba:

```bash
mamba install -c conda-forge tectonic pygments
```

Build from the repository root:

```bash
make
```

The first Tectonic run may download its TeX bundle. If the default cache
directory is not writable, the root Makefile already points Tectonic at a
writable `/tmp` cache by default. Override it if desired:

```bash
make TECTONIC_CACHE=$HOME/.cache
```

The generated PDF will be:

```bash
slides/build/main.pdf
```

The slides are narrative only. Runnable CuTe DSL examples live in the chapter
directories and are linked from the deck.

The preamble includes `minted` for code snippets. The root Makefile enables
Tectonic shell escape with `-Z shell-escape`, which `minted` needs in order to
call `pygmentize`.
