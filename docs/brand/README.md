# Brand assets

`aithne-logo-source.png` is the badge the logo came from — 153 × 119, and the
only original. Everything the app ships was traced from it:

| File | What it is |
| --- | --- |
| `frontend/src/icons.tsx` → `AithneMark` | The mark as an inline path. Fills with `currentColor`, so the rail tints it sage and a light page takes `--brand-mark`. |
| `frontend/public/aithne-mark.svg` | The same path, fixed at the logo green `#2eaa3c`, for anything that links to an image rather than inlining one. |
| `frontend/public/favicon.svg` | The mark on the badge's white oval, so it holds up against a dark browser chrome. |
| `frontend/public/aithne-icon-180.png`, `-512.png` | Square raster icons, for `apple-touch-icon` and anywhere a PNG is required. |

To change the logo, replace the source PNG and re-trace; do not hand-edit the
path data.
