# AI Mine Design System

An open-pit resource and pit-limit pipeline that replicates the Surpac workflow end to
end: **collar → survey → assay → geology → 3D → orebody solid → composited/top-cut
assays → block model → ultimate pit limit → report**.

Four source tables go in — CSV or Excel, mixed freely, or one workbook.
One self-contained HTML report comes out, with an interactive 3D model of the whole
resource: drillholes, the orebody solid, the block model and the ultimate pit.

**Exactly two LLM calls per run, regardless of dataset size.** Everything else —
desurveying, triangulation, statistics, compositing, variography, kriging,
point-in-solid tests, tonnage and pit optimization — is deterministic Python with a
known formula. Routing those through a model would cost money and add a hallucination
risk where none is needed.

---

## Design philosophy

Only two steps in this pipeline require genuine judgement over ambiguous, conflicting
signals:

| Call | Decides | Model |
|---|---|---|
| **1. Orebody & Grade Modeling** | ore/waste cutoff, domain boundary per section, variogram model and estimation method | `MODEL_OREBODY_GRADE` |
| **2. Pit Design & Report** | pit economics, slope assumptions, the written engineering narrative | `MODEL_PIT_REPORT` |

Everything else is arithmetic. That split is enforced in the code, not just documented:
the model writes prose and picks parameters, and the pipeline supplies every number that
appears in a table, a chart or a 3D layer. No figure in the report can be a fabrication.

If neither the paid model nor the local fallback is reachable, the run still completes
using coded geostatistical conventions — and the report says so at the top, in a banner,
rather than quietly presenting a convention as an engineering judgement.

---

## Quick start

```bash
python -m pip install -r requirements.txt
cp .env.example .env          # then add your HF_TOKEN

python scripts/make_sample_data.py     # generate a synthetic 66-hole gold deposit
python run_pipeline.py --sample        # run the whole workflow
```

The report lands in `outputs/`. Open it in any browser — it needs no network.

To run without spending anything on API calls:

```bash
python run_pipeline.py --sample --offline
```

### Your own data

```bash
python run_pipeline.py \
    --collar data/collar.csv --survey data/survey.csv \
    --assay data/assay.csv --geology data/geology.csv \
    --project "Mount Isa South" \
    --price 2400 --mining-cost 2.80 --recovery 91 --slope 48
```

Or from one Excel workbook (a sheet per table — see [Input schema](#input-schema)):

```bash
python run_pipeline.py --workbook data/deposit.xlsx --project "Mount Isa South"
```

Any economic value you pass is treated as **fact** and is never overridden by the model;
anything you omit is proposed by Call 2 and labelled `assumed_default` in the report's
assumptions table.

### As a service

```bash
uvicorn minedesign.api:app --reload
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness and which model paths are configured |
| `POST /run` | run on four server-side file paths |
| `POST /upload` | run on four uploaded files (CSV or Excel) |
| `GET /reports` | list generated reports |
| `GET /reports/{name}` | fetch one report |

---

## Input schema

Column names are matched against a synonym dictionary covering Surpac, Datamine,
Micromine and Vulcan conventions, so `BHID`/`HoleID`/`hole_id` and `East`/`Easting`/`X`
all resolve. Only a column the dictionary cannot resolve escalates to the local
fallback model, and only for the file that failed.

**Formats**: CSV, Excel (`.xlsx`/`.xlsm`/`.xls`), and WAMEX/GSWA statutory `.txt`
lodgement files, detected by content rather than by extension.

### WAMEX / GSWA files

Western Australian exploration data is lodged in a fixed tab-separated layout:
metadata lines keyed `H0002`-`H0900`, column names on `H1000`, units on `H1001`,
analytical method on `H1002`, data rows prefixed with a literal `D`, and a
trailing `EOF`. Three details in that format are handled explicitly, because
each one fails *silently* otherwise:

- **Repeated column names.** An assay file carries the same element once per
  analytical method. Letting pandas rename the second `U` to `U.1` discards the
  one fact that separates them, so the method line names them instead:
  `U_4AMS`, `U_WACMS`, `U_WAPMS`.
- **Censored laboratory values.** `<0.5` and `>2000` turn an entire element
  column into text, and every grade then becomes NaN. Below-detection results
  are substituted with half the detection limit and over-range results with the
  limit itself — the standard convention — and both counts are reported.
- **No column called "grade".** A multi-element file offers dozens of
  candidates, and the synonym dictionary would bind the first one it recognises.
  On a uranium project that is `Cu`. So the element must be stated via
  `grade_element` / `--grade-element`, and an unstated one raises with the list
  of available elements rather than guessing. When that element was assayed
  several ways, a total digest (4-acid, fusion) is preferred over a weak-acid
  partial leach, which recovers only a fraction of the contained metal, and the
  column actually used is recorded in the report.

| File | Required | Optional |
|---|---|---|
| collar | hole id, easting, northing, elevation | max depth |
| survey | hole id, depth, azimuth, dip | |
| assay | hole id, from, to, grade | |
| geology | hole id, from, to, rock type | |

Conventions: azimuth is degrees clockwise from grid north, dip is **negative
downwards** (all-positive dip files are detected and normalised, with a flag), and `z`
is elevation increasing upwards.

### CSV or Excel, any mix

Each of the four inputs may be `.csv` **or** `.xlsx`/`.xlsm`/`.xls` — the loader
dispatches on file extension, so nothing else in the pipeline knows or cares which one
it got. All four may also point at the **same Excel workbook**, one sheet per table:

```bash
python run_pipeline.py --workbook data/deposit.xlsx --project "Mount Isa South"
```

The sheet for each table is picked by matching its name against `collar`/`survey`/
`assay`/`geology` (`"Collar Data"`, `"COLLAR"`, `"collar_2024"` all match); a workbook
with only one sheet, or no matching name, falls back to the first sheet. Sources can be
mixed — `--workbook` fills in whatever `--collar`/`--survey`/`--assay`/`--geology` you
leave unset, so a workbook plus one corrected CSV works too. `.xls` (the legacy binary
format) needs the optional `xlrd` package in addition to `openpyxl`.

---

## What the pipeline does

```
[code] ingest → validate → desurvey → colour → composite → top-cut
   → [LLM CALL 1: cutoff + domain boundaries + variogram/method]
   → [code] triangulate → validate_solid (retry: local fallback, max 1x)
   → [code] block grid → estimate → point_in_solid → tonnage
   → [code] preliminary pit + nested shells at default economics
   → [LLM CALL 2: pit economics + report narrative]
   → [code] final ultimate pit → scene, charts, tables → HTML

   [code] GemPy implicit orebody model runs alongside, from the same ore flags,
          with no LLM in the loop at any point
```

Notable implementation choices:

- **Desurvey** is minimum curvature, implemented directly from the standard ratio-factor
  formula.
- **Compositing** is fixed-length and length-weighted, at the modal raw sample length.
  The top cut defaults to the 99th percentile and reports what share of contained metal
  the cap removes — a cap that strips a lot of metal is the signal to revisit it.
- **The solid** is built by lofting section polygons resampled in polar coordinates, so
  corresponding vertices line up and the loft cannot twist. It is validated for
  watertightness, connectivity, duplicates, non-manifold edges, self-intersection and
  consistent winding. Volume is verified against an analytic cylinder in the test suite
  to within 0.3%.
- **Estimation** is ordinary kriging with a moving search neighbourhood, or IDW when the
  data density does not support kriging. Leave-one-out cross validation is reported.
- **The ultimate pit** is the maximum-weight closure of the block precedence graph.
  Lerchs and Grossmann solved that with their graph algorithm in 1965; Picard showed in
  1976 that the same closure comes exactly from a minimum cut, which is what runs here —
  the true optimum, via a fast C min-cut. The test suite checks it against brute-force
  enumeration and verifies the result is a valid closure.
- **Blocks outside the wireframe are waste**, whatever the estimator interpolated into
  them. Without that constraint the optimizer sells grade smeared into the padding and
  reports more ore in the pit than the deposit contains.

### A caveat on slope angles

Pit slopes are enforced with a multi-bench arc template. Achievable angles are quantized
by the block size: on 10 m blocks, 45° and 63.4° are exact, but a requested 40° is cut at
41.8°. **A steeper wall than requested understates waste and flatters the pit**, so the
report always states the achieved angle next to the requested one and warns when they
differ by more than 1.5°. Use a smaller block size or an achievable angle if it matters.

---

## Pit and bench design (report section 10)

The ultimate pit optimization produces an economic **shell** — the maximum-value set of
blocks, with no benches or catch berms defined. This section slices that shell into a real
per-bench schedule: bench count, bench height, RL, footprint and volume, bench by bench,
plus the minimum footprint half-width around 8 compass sectors for every bench — the
geometric quantity a batter/berm design is actually constrained by, since a bench pinched
narrow in one direction can't carry a full berm there even if its average width is
generous.

Every number in the schedule traces back to the block model exactly — footprint areas are
exact cell counts (not a smoothed contour estimate), and the total volume matches the real
rock in the optimizer's own result to machine precision, verified by a regression test.

**Getting the crest right took a real bug fix.** Lerchs-Grossmann's precedence constraint
pulls "air" blocks — grid cells above topography, worth nothing — into the closure
whenever real material beneath them is worth mining, since including them costs nothing.
An early version of this schedule inferred the pit's crest from the raw in-pit block
z-range, which counted that overhead air as if it were rock to excavate: a 37% tonnage
overstatement on the deposit it was first caught on, with no error raised, only a wrong
number. The fix computes two grids, not one — each column's floor **and** its real-rock
crest (air explicitly excluded) — and a bench only counts a column when rock spans the
bench's full height, not just its floor.

**No bench stability figure is stated, on purpose.** A real batter angle, catch-berm
width, or Factor of Safety is a geotechnical design output that needs data this pipeline
was never given: rock mass rating, discontinuity orientations, shear strength, groundwater
conditions. What the schedule does show is the geometric relationship any stability design
has to satisfy — `overall_slope = atan(bench_height / (bench_height/tan(batter) + berm))`
— against the verified overall slope the pit was actually optimized at, and flags the
narrowest bench as the first place a real stability review should look.

---

## The GemPy model (report section 12)

The report carries **two independent orebody models**, built from the same drilling by
methods that share no code and no assumptions.

| | Section wireframe | GemPy model |
|---|---|---|
| Kind | explicit | implicit |
| How | ore envelopes digitised per drill section, lofted between sections | a scalar potential field interpolated through the ore contacts |
| Shape comes from | the digitised outlines | the data and the interpolation |
| LLM involved | yes, Call 1 picks the cutoff and boundary points | **no, none at any point** |

GemPy interpolates a potential field through the observed ore contacts, constrained by
orientations fitted to those same contacts, and takes the orebody as an isosurface of
that field. The result is reported next to the wireframe volume, because two methods
this different agreeing is evidence, and disagreeing is a question worth asking. On the
bundled sample deposit they land within **4.3%** of each other.

Skip it with `--no-gempy`; the pipeline runs without GemPy installed and the section then
states why it is empty.

### The modelling choice that matters

An orebody must be modelled as **one closed contact surface with outward-pointing
normals**, not as a hanging wall and a footwall as two separate surfaces. Two surfaces in
one GemPy series are isosurfaces of the same field and so are forced roughly parallel —
but an orebody is a closed body whose top is a dome and whose base is a bowl. Modelled as
a pair, the field compromises between contradictory orientations and the two surfaces
collapse onto each other. In testing that failure produced a solid holding **9.6%** of the
ore intercepts; the closed-surface formulation holds **95%**.

Because the normals must reverse between the top and the bottom of the body, each one is
fitted to the nearest contacts and then turned to face away from the local centre of the
ore. The report plots them on a rose so that reversal is visible.

### Capture rate

The report leads with the share of logged ore intercepts that fall inside the implicit
solid. A potential field always produces a smooth, convincing surface; whether it
actually contains the ore is a separate question, and that number is what answers it.

### What GemPy does not settle

It interpolates; it does not know geology. There is no fault or vein-set structure, no
domain boundary beyond the contacts it is given, and it extrapolates confidently into
ground with no drilling. It also inherits the ore/waste flag it is handed — so the cutoff
defining a contact still comes from Call 1. What GemPy removes from the loop is the
**shape**, not the threshold.

---

## The report

One HTML file, no network needed — three.js, its orbit controls and Chart.js are vendored
into the package and inlined at render time.

**One 3D scene**, orbitable, with every spatial figure as a toggleable layer: drillholes
(switchable between grade and rock-type colouring), section boundaries, the orebody
solid, raw samples, composites, top-cut composites, the block model, the pit blocks and
the pit shell, plus topography. Section view locks the camera to a single drill section,
matching a conventional cross-section.

The pit shell is drawn as elevation-banded contour rings rather than one flat surface —
the same purpose as digitizing a series of "quick planes" through a solid at a fixed
interval, which is how a pit shell is normally read level by level rather than as one
undifferentiated mesh. A column belongs to the pit at elevation *E* exactly when its
floor RL is at or below *E*, a direct consequence of the optimizer's precedence
constraint (a block is only ever mined if the block above it is too); that turns the pit
floor grid already computed for tonnage into a plan-view mask at any elevation, which is
what lets the ring spacing be chosen independently of the block model's own bench height.

**2D charts** for everything statistical, because forcing a distribution into 3D makes it
harder to read: grade histogram with a fitted lognormal curve, cumulative frequency,
top-cut before/after, experimental variograms with fitted models, the grade-tonnage
curve, and nested pit shells by revenue factor.

**Tables**: solid validation report, descriptive statistics before and after top cut, the
full composited top-cut file and the full block model — the last two searchable and
scrollable.

---

## Configuration

All configuration is environment-driven; see `.env.example` for the annotated template.

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face Inference Providers token |
| `HF_INFERENCE_BASE_URL` | OpenAI-compatible router base URL |
| `MODEL_OREBODY_GRADE` | model for Call 1 |
| `MODEL_PIT_REPORT` | model for Call 2 |
| `LOCAL_FALLBACK_MODEL` | Ollama model, free, used only on failure |
| `LOCAL_LLM_BASE_URL` | Ollama endpoint |
| `MINEDESIGN_OFFLINE` | `1` skips both paid calls |
| `DEFAULT_DENSITY` | in-situ bulk density used when none is supplied |

The local fallback is hit in exactly two situations: a CSV column the dictionary could
not map, and a wireframe that failed validation twice. It never makes the primary
decisions.

---

## Tests

```bash
python -m pytest tests/ -q
```

The suite checks maths against known answers rather than against previous output:
desurvey against trigonometry, solid volume and surface area against an analytic
cylinder, point-in-solid against Monte Carlo integration, kriging weights against the
unbiasedness property, and the pit optimizer against brute-force enumeration plus an
explicit closure check.

---

## Honest caveat

This is a decision-support tool, **not** a certified replacement for Surpac in JORC or
NI 43-101 reporting, which requires auditable methodology and Competent/Qualified Person
sign-off.

Have a mining engineer review the cutoff-grade, variogram-method and pit-economics
decisions before using any output for real decisions — those are exactly the two calls
where judgement, and error, live.

---

## Layout

```
minedesign/
  config.py            environment-driven settings
  ingest/              column mapping, loading, schema and quality checks
  geology/             desurvey, statistics, compositing, top cut, colouring
  solid/               sections, triangulation, validation, point-in-solid
  blockmodel/          grid, topography, variography, kriging, tonnage
  pit/                 ultimate pit limit by maximum closure
  gempy_model/         implicit orebody model: contacts, interpolation, sections
  llm/                 transport, prompts, the two agent calls
  report/              3D scene, charts, HTML renderer, vendored JS
  pipeline.py          orchestration
  api.py               FastAPI service
scripts/               synthetic dataset generator
tests/                 known-answer test suite
```
