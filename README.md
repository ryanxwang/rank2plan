# rank2plan

Implementation of constraint generation and column generation (soon) for solving
large L1-RankSVMs with hinge loss (with pair-specific gaps) and sample weights.
This is based on the work by Dedieu et al (2022) on solving large L1-SVMs with
hinge loss. See `documents/theory.pdf` for how we extend their work. The "2plan"
part of the package name comes from the tool being used to learn heuristics for planning.

## References

- A. Dedieu, R. Mazumder, and H. Wang. Solving l1-regularized svms and related
linear programs: Revisiting the effectiveness of column and constraint
generation. J. Mach. Learn. Res., 23:164:1–164:41, 2022. [[URL]](http://jmlr.org/papers/v23/19-104.html).
