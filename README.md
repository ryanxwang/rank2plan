# rank2plan

[![Tests](https://github.com/ryanxwang/rank2plan/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/ryanxwang/rank2plan/actions/workflows/pytest.yml)

Implementation of constraint generation and column generation for solving large
L1-RankSVMs with hinge loss (with pair-specific gaps) and sample weights. This
is based on the work by Dedieu et al (2022) on solving large L1-SVMs with hinge
loss. See `documents/theory.pdf` for how we extend their work. The "2plan" part
of the package name comes from the tool being used to learn heuristics for
planning.

## Installation

Install with

```bash
pip install rank2plan
```

The PyPI release only supports `3.9 <= python` for now.

## Examples

See under `tests` for examples.

## Todo

- [ ] Support for sparse matrices and vectors
- [x] Bayesian optimisation for tuning the `C` value.
- [ ] We log pretty aggressively, probably should add a verbosity control

## References

- A. Dedieu, R. Mazumder, and H. Wang. Solving L1-regularized SVMs and Related
Linear Programs: Revisiting the Effectiveness of Column and Constraint
Generation. J. Mach. Learn. Res., 23:164:1–164:41, 2022. [[URL]](http://jmlr.org/papers/v23/19-104.html).
