# Report template source

The unchanged `aaai2026.sty` and `aaai2026.bst` files come from the CameraReady/
LaTeX directory in the official AAAI 2026 author kit:

https://aaai.org/wp-content/uploads/2025/07/AuthorKit26-1.zip

Retrieved November 3, 2026. Archive SHA-256:
`d2844ec68a4a9396d749fcca5b5784809617b670863e8d4fecbfb00e444fc3af`.

The course report uses the style's `draft` option to show the named author
without claiming AAAI publication or copyright transfer. A small engine
compatibility block in `report.tex` supports XeTeX/Tectonic; the style itself
is unmodified. The professor's handout imposes an strictly fewer than eight pages.
The earlier course brief requests AAAI formatting, retained here.

The report's result macros are generated from completed experiments. Never
fill them with hypothetical or anticipated results. Build from this directory
using `tectonic report.tex`, or a standard LaTeX/BibTeX toolchain.
