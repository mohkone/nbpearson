# Provenance and third-party notices

The numerical implementation is the evaluated nbpearson 0.1.0.dev1 five-module package. Historical extraction and source-hash records remain separately preserved. Release packaging does not change the implementation or erase provenance.

Mohamed Kone explicitly asserted ownership of the original nbpearson source code and authority to distribute it under MIT. On that basis, `LICENSE` contains the complete standard MIT text with `Copyright (c) 2026 Mohamed Kone`. The year is the intended initial release-notice year, not a claim about historical development dates. This is an attributed investigator assertion; no institutional waiver or approval by other manuscript authors is inferred.

NumPy and SciPy are external runtime dependencies. Their source, binaries, bundled libraries and site-packages are excluded and retain their own licences/notices. nbpearson's MIT metadata does not relicense dependencies. Consult actual dependency distributions for applicable notices. Build tools and Python likewise retain their own terms.

The code calls SciPy's BarycentricInterpolator, LinearOperator, sparse interfaces and numerical routines. Examples may use SciPy svds. Classical interpolation, rationalized differences, compensated operations, sparse-plus-low-rank products, exact-depth grouping and iterative SVD are established techniques; packaging is not a mathematical priority claim.

Bundled examples and autonomous tests use synthetic counts. No biological matrix, benchmark archive, manuscript, private author record or experiment runtime is included. The MIT notice applies to original nbpearson material within the investigator's asserted authority; it does not license 10x/GEO datasets, BPCells or other third-party code. BPCells is not bundled and is not a runtime dependency. Dataset redistribution rights remain separate.

The standard MIT wording is retained unchanged from candidate V1, whose record documents consultation of the Open Source Initiative's [MIT licence](https://opensource.org/license/mit); V2 fills only the normal copyright fields.
