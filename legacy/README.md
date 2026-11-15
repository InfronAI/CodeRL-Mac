# Original CodeRL implementation

The Python entry points, configurations, dataset classes, trainer and shell scripts in this directory are the unmodified original code from upstream commit `ec3c526a07e44f56f702f354cbad11187bdf25cc`, moved together to avoid colliding with modern packages. `README.upstream.md` is the original README. Its relative paths describe the original repository layout.

The original customized Transformers fork remains at `../transformers/`; the original data and outputs remain at `../data/` and `../outputs/`. The new M1 runtime does not install or import this code. Use the root README and `coderl` command for supported local execution.

The legacy environment pins PyTorch 1.9.1 and uses CUDA/DeepSpeed assumptions. Merely running these files from their new directory does not make that old environment M1 compatible. They are retained as reference and for provenance, not presented as tested modern entry points.
