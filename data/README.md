# Kestrel Outfitters data

This folder holds the running case-study data for *Data Analysis with Python*.

## Files

| File | Rows | Size | Notes |
|---|---|---|---|
| `raw/kestrel_orders.csv` | 489,120 | ~41 MB | Order-line export, one row per line on an order. Synthetic; safe to redistribute. |
| `raw/kestrel_customers.csv` | 42,868 | ~1.5 MB | Customer dimension: `signup_date`, `loyalty_tier`, `home_region`. Chapter 3 join example. |
| `raw/kestrel_products.csv` | 40 | ~2 KB | Product dimension: `product_name`, `category`, `list_price`. Chapter 3 join example. |

`processed/` is created by the chapter code (for example Step 10 writes
`orders_clean.csv`). Raw files are never edited in place.

The two dimension files are generated deterministically from the orders file by
`scripts/generate_kestrel_dimensions.py`. They deliberately omit 50 customers
who appear in the orders file, so the Chapter 3 left-join indicator has real
`left_only` rows.

## Chapter 4 acquisition sources

Chapter 4 reads data from many containers, so it ships a set of generated
fixtures plus one small local HTTP server. Everything is reproducible offline:

```bash
python scripts/generate_chapter04_sources.py   # writes the files below
python scripts/mock_kestrel_api.py             # local API + catalogue server
python scripts/chapter04_outputs.py            # prints every output block
```

| Path | What it is | Used for |
|---|---|---|
| `raw/sales_exports/sales_2024_01.csv` … `_12.csv` | 12 monthly exports (2,573 rows total) | Schema-drift consolidation (§4.8) |
| `raw/eu_orders.csv` | Semicolon + decimal comma, cp1252 accents | Delimiter/encoding (§4.2) |
| `raw/contacts_utf8_bom.csv` | UTF-8 with a byte-order mark | BOM handling (§4.2) |
| `raw/kestrel_q1_2026.xlsx` | Merged title row, header on row 3, cached `=SUM` | Excel reading (§4.2) |
| `raw/campaigns.jsonl` | JSON Lines, 5 rows | `lines=True` (§4.2) |
| `raw/store_payload.json` | Nested store object | `json_normalize` (§4.2) |
| `raw/legacy_products.txt` | Fixed-width text | `read_fwf` (§4.2) |
| `raw/store_orders.xml` | Nested order/line XML | `ElementTree` (§4.2) |
| `raw/exports_archive.zip` | Three regional CSVs | Archive reads (§4.2) |
| `raw/catalogue/page_1.html`, `page_2.html`, `product_*.html` | 20-product catalogue | Scraping (§4.5) |
| `raw/weather/open_meteo_chicago.json` | Cached Open-Meteo response | One real API call (§4.4) |

`raw/kestrel.db` is a SQLite database built from the orders, customers, and
products files (orders 487,922 / customers 42,868 / products 40 / stores 12).
It is about 64 MB and is **generated**, not committed—regenerate it with
`scripts/generate_chapter04_sources.py`. The same applies to the files above;
they are all deterministic outputs of that script.

The mock server serves `/v1/orders` (offset pagination), `/v1/orders/cursor`,
`/v1/secret` (bearer auth), `/flaky` (429 then 200), `/catalogue`, and
`/exports/<file>.csv`, so the API and scraping examples need no network or key.

## Chapter 5 audit fixtures

Chapter 5 inspects and profiles data, so it ships a small set of deliberately
defective tables alongside the clean feed. Everything is deterministic:

```bash
python scripts/generate_chapter05_sources.py   # writes the files below
python scripts/chapter05_outputs.py            # prints every output block
```

| Path | What it is | Used for |
|---|---|---|
| `raw/audit/orders_sample_dirty.csv` | 31 order lines with ~23 planted defects | Validation suite (§5.6), Example Bank (§5.11) |
| `raw/audit/customers_dirty.csv` | 25 messy customer records | Second scorecard (§5.5.4), Mini-Project 2 |
| `raw/audit/finance_monthly_net.csv` | 27 months of net revenue, one manual accrual | Reconciliation (§5.8), Mini-Project 3 |

The dirty orders sample plants: an exact duplicate row, a duplicate key with a
conflicting quantity, a missing key, an unparseable date, an impossible date
(`2026-02-30`), a future date, a blank region, a non-breaking-space region, a
case-variant and a whitespace-padded category, `-1` and `9999` sentinels, a
negative, a zero, and a currency-formatted price, an out-of-range discount, an
out-of-domain `returned`, two broken foreign keys, a cross-field revenue
mismatch, and a negative shipping charge.

`finance_monthly_net.csv` is derived from `kestrel.db` as
`gross - returns - shipping` per month, with a single deliberate `+15000`
"manual accrual reversal" in 2026-02, so the reconciliation has a real,
explainable difference to find.

## Chapter 6 cleaning fixtures

Chapter 6 repairs a purpose-built messy feed. Everything is deterministic:

```bash
python scripts/generate_chapter06_sources.py   # writes the files below
python scripts/chapter06_outputs.py            # prints every output block
```

| Path | What it is | Used for |
|---|---|---|
| `raw/audit/orders_messy_40k.csv` | 40,000 order lines with 14 planted defect types | Whole chapter, §6.2–§6.11 |
| `raw/audit/customers_messy.csv` | 2,300 messy customer records across 950 households | Fuzzy household merge (§6.4.5) |
| `raw/audit/households_truth.csv` | `record_id` → true `household_id` | Scoring the merge (held out from the rules) |
| `raw/audit/income_churn.csv` | 3,000 customers, income 12.13% missing | Missing-data mechanisms and imputation (§6.1, §6.3) |
| `raw/audit/fx_rates.csv` | 195 weekly rates to USD for EUR/GBP/CAD | As-of currency join (§6.6.3) |
| `raw/audit/eu_orders_fx.csv` | 600 multi-currency orders | As-of currency join (§6.6.3) |

The orders fixture is sampled from rows of `kestrel_orders.csv` that are
already clean, so every defect in it is planted and countable. The fourteen
defect families are: blank values (including a 100-row block where region,
discount, and customer ID are blank together), sentinels (`-1`, `9999`,
`unknown`, `N/A`, `1900-01-01`), exact duplicates (250), conflicting business
keys (120), case variants (200), whitespace and non-breaking spaces (200),
currency-formatted prices (150), negative quantities (80), zero prices (60),
bad dates (100 impossible or unparseable plus 40 future), mixed US-style date
formats (120), out-of-range discounts (100), broken foreign keys (90), and
stored revenues that do not recompute (60). The fixture is written to text so
the sentinels and non-breaking space survive round-tripping.

`customers_messy.csv` contains name variants (nicknames, initials, case,
one-character typos), address variants (`Street`/`St`, lower case, apartment
numbers), shared-family emails, several phone formats, and fifteen pairs of
roommates at shared addresses with different surnames. The roommates are the
false-positive trap for a naive address-only merge.

`income_churn.csv` plants MAR missingness: the probability that income is
missing depends only on the segment (Basic 16%, Plus 10%, Premium 5%), while
churn is approximately flat across segments. A stratified permutation test of
churn against the missing flag within segment is non-significant
(p = 0.268), supporting a segment-wise median as the MAR-aware treatment.

Generated outputs are written to `data/processed/orders_clean_ch06.csv`,
`data/processed/orders_exceptions_ch06.csv`, and `reports/`. The SHA-256
prefix of `orders_messy_40k.csv` is `71739b6ceb52` (3,554,509 bytes).

## Chapter 7 join and reshape fixtures

Chapter 7 assembles the analysis table. It reads the existing orders, customer,
product, and store files plus the twelve monthly exports, and adds five small
fixtures:

```bash
python scripts/generate_chapter07_sources.py   # writes the files below
python scripts/chapter07_outputs.py            # prints every output block
```

| Path | What it is | Used for |
|---|---|---|
| `raw/audit/price_history.csv` | 120 effective-dated prices for 40 products | As-of join (§7.5) |
| `raw/audit/customers_join_dirty.csv` | 100 keys: twenty customers in five spellings | Key normalization (§7.3.5) |
| `raw/audit/customers_duplicate_keys.csv` | 62 rows / 50 keys | Fan-out trap (§7.4) |
| `raw/audit/monthly_report_wide.csv` | 4 regions × 36 metric-month columns | Reshaping round-trip (§7.8) |
| `raw/audit/campaign_customer.csv` | 1,200 campaign exposures with spend | Customer 360 (§7.10) |

`customers_duplicate_keys.csv` is the key one: joining it to orders turns
20,000 lines into 22,567 and inflates revenue by $230,216.96, which is the
chapter's worked fan-out example. `monthly_report_wide.csv` melts to 144 rows
(4 regions × 36 columns) and rebuilds exactly.

Generated outputs are written to `data/processed/customer360_ch07.parquet`
and `reports/ch07_monthly_scorecard.csv`.

## Chapter 8 EDA fixtures and figures

Chapter 8 explores a prepared customer table and ships four fixtures plus nine
figures. Everything is deterministic:

```bash
python scripts/generate_chapter08_sources.py   # fixtures + images/ch08_*.png
python scripts/chapter08_outputs.py            # prints every output block
```

| Path | What it is | Used for |
|---|---|---|
| `raw/audit/churn_eda.csv` | 3,000 customers, 34 mixed-type columns | Whole chapter |
| `raw/audit/anscombe_quartet.csv` | four datasets with identical summaries | Checkpoint |
| `raw/audit/simpson_segments.csv` | aggregated two-segment reversal | Simpson's paradox (§8.7) |
| `raw/audit/distribution_gallery.csv` | 2,000 rows of six distribution shapes | Shape section (§8.4) |

The churn table has a 12.5% missing income tied to segment, a bimodal
`session_minutes` column, a zero-inflated `support_tickets` column, and a
long right tail in `total_spend` (mean $309.30, median $213.77). The Simpson
fixture shows control winning overall (80.3% vs 75.8%) while treatment wins
inside both segments.

Nine figures are written to `images/ch08_*.png`: histogram/ECDF/box plot,
distribution gallery, correlation heatmap, pair plot by segment, bootstrap
group medians, Simpson reversal, Anscombe quartet, Q–Q plots, and a signup
cohort trend. Figures are referenced from the chapter with `image::` macros.

## Chapter 9 inference fixtures

Chapter 9 builds its examples from four deterministic fixtures and six figures:

```bash
python scripts/generate_chapter09_sources.py   # fixtures + images/ch09_*.png
python scripts/chapter09_outputs.py            # prints every output block
```

| Path | What it is | Used for |
|---|---|---|
| `raw/audit/ab_checkout.csv` | 12,000 users, control/treatment conversion, guardrails | A/B test (§9.10, §9.13) |
| `raw/audit/order_values_ch09.csv` | 5,000 right-skewed order values | CLT, bootstrap, t-intervals (§9.6, §9.7) |
| `raw/audit/sampling_frame.csv` | 50,000 customers with a known mean | Sampling designs and bias (§9.5) |
| `raw/audit/paired_campaign.csv` | 800 before/after spend pairs | Paired tests (§9.9) |

The checkout test produces 12.33% conversion in control and 13.00% in
treatment: +0.66 percentage points, p = 0.276, with a support-contact
guardrail that rises from 2.72% to 3.37%. The intended lesson is a **do not
roll out** decision, not a win. The sampling frame has a known mean of
$62,885.40; self-selected respondents average $73,727.68, a $10,842.28 bias
that no confidence interval captures.

Six figures are written to `images/ch09_*.png`: sampling distribution,
bootstrap median, interval coverage, power curve, sampling-bias comparison,
and A/B conversion with intervals.

## Chapter 10 regression fixtures

Chapter 10 measures relationships with four deterministic fixtures and seven
figures:

```bash
python scripts/generate_chapter10_sources.py   # fixtures + images/ch10_*.png
python scripts/chapter10_outputs.py            # prints every output block
```

| Path | What it is | Used for |
|---|---|---|
| `raw/audit/ad_spend.csv` | 365 days of spend, revenue, weekend, channel | Simple regression and diagnostics (§10.4, §10.5) |
| `raw/audit/housing_listings.csv` | 1,200 listings with correlated predictors | Multiple regression, VIF, interaction (§10.6) |
| `raw/audit/churn_regression.csv` | 3,000 customers with a logistic outcome | Logistic regression (§10.7) |
| `raw/audit/correlation_cases.csv` | six constructed relationships | Correlation traps (§10.2) |

The daily spend model gives a slope of $2.34 of revenue per dollar of spend
(95% CI 2.19 to 2.49), R² 0.726, and heteroscedastic residuals (Levene
p = 0.0039). In the housing data `size_sqft` and `rooms` correlate at 0.989
with VIF about 47; dropping `rooms` cuts the size standard error sevenfold.
The churn model has AUC 0.867 with well-calibrated quintiles.

Seven figures are written to `images/ch10_*.png`: simple regression fit,
four diagnostic panels, ordered correlation heatmap, interaction slopes,
ROC and calibration, four correlation shapes, and lasso coefficient paths.

## Chapter 11 visualisation fixtures

Chapter 11 builds its figures from four deterministic fixtures and twelve
PNG files:

```bash
python scripts/generate_chapter11_sources.py   # fixtures + images/ch11_*.png
python scripts/chapter11_outputs.py            # prints every output block
```

| Path | What it is | Used for |
|---|---|---|
| `raw/audit/viz_sales.csv` | 288 rows: 24 months x 4 regions x 3 categories | Whole chapter |
| `raw/audit/viz_distributions.csv` | 1,500 rows of five distribution shapes | Histograms, ECDF, box/violin |
| `raw/audit/viz_cohort.csv` | 36-cell retention triangle | Cohort heatmap |
| `raw/audit/viz_bad_chart.csv` | 12 nearly equal categories | Chart makeovers |

Every figure is drawn with Matplotlib, which is installed in this project.
Seaborn and Plotly are optional companions: their sections show the real APIs
as guarded examples, and a clear message is printed when the package is
absent. The static equivalents cover the same data and design decisions.

The sales table totals $11,646,999.12 of revenue, 61,103 orders, and
$2,897,643.92 of margin. A truncated six-month axis amplifies the apparent
change 4.4 times; radius-scaled bubbles misstate a 7.7x value ratio; and the
last three months fell 11.5% while the full 24 months rose 15.5%.

Twelve figures are written to `images/ch11_*.png`: four-panel performance,
axis comparison, grouped bars with error bars, small multiples, annotated
line, box/violin/strip, correlation heatmap, bad pie, bubble scaling, cohort
triangle, chart makeover, and final story chart.

## Chapter 12 time-series fixtures

Chapter 12 builds its analysis from four fixtures and six figures. The
classical routines that would normally come from statsmodels are implemented
here with NumPy and pandas, because statsmodels is not installed:

```bash
python scripts/generate_chapter12_sources.py   # fixtures + images/ch12_*.png
python scripts/chapter12_outputs.py            # prints every output block
```

| Path | What it is | Used for |
|---|---|---|
| `raw/audit/ts_monthly.csv` | 48 months x 3 channels, trend + December peak | Whole chapter |
| `raw/audit/ts_daily.csv` | 730 days of sessions, weekly cycle, 3-day outage | Resampling, smoothing, anomalies |
| `raw/audit/ts_dates_messy.csv` | three date formats plus DST cases | Parsing and time zones (§12.1) |
| `raw/audit/ts_events.csv` | four logged events | Anomaly investigation (§12.10) |

The monthly series ends at $397,163 in December 2025, up 8.1% year over year,
with a multiplicative December index of 1.33. Rolling-origin MASE averaged
2.93 for naive, 0.87 for seasonal naive, and 0.52 for Holt–Winters. The naive
80% interval covered only half of the backtest points, so it needs recalibration.

Six figures are written to `images/ch12_*.png`: decomposition, ACF/PACF,
rolling averages, forecast with band, the outage anomaly, and a scenario fan.

## Chapter 13 SQL fixtures

Chapter 13 queries a small retail database:

```bash
python scripts/generate_chapter13_sources.py   # rebuilds shop.db and extracts
python scripts/chapter13_outputs.py            # prints every query result
```

| Path | What it is | Used for |
|---|---|---|
| `raw/shop.db` | SQLite database, 2 MB | Whole chapter |
| `raw/shop_customers.csv` | 2,000 customers | pandas cross-validation |
| `raw/shop_orders.csv` | 8,000 orders | pandas cross-validation |
| `raw/shop_order_items.csv` | 11,431 lines | pandas cross-validation |

Tables: `customers` (2,000), `orders` (8,000), `order_items` (11,431),
`products` (36), `campaigns` (4), `exposures` (3,200), `targets` (96),
`funnel_events` (16,022), `staging` (4), plus `customers_dup` (2,012) and the
`v_order_lines` view. The duplicate dimension and the messy staging table are
deliberate: joining to `customers_dup` adds 47 rows and $16,488.91 of invented
revenue. The clean join totals $3,163,258.68.

Other environments are available in the book's toolchain: `sqlalchemy` and
`duckdb` are installed, but the chapter uses the standard `sqlite3` module so
every query runs without an extra dependency.

## Chapter 14 Excel fixtures

Chapter 14 audits a deliberately messy workbook and generates two reports:

```bash
python scripts/generate_chapter14_sources.py   # writes the workbooks below
python scripts/chapter14_outputs.py            # prints every output block
```

| Path | What it is | Used for |
|---|---|---|
| `raw/excel/sales_workbook.xlsx` | 7 sheets, 200 data rows, 202 formulas | Whole chapter |
| `raw/excel/monthly_sales.xlsx` | 12 monthly worksheets | Folder import (§14.8) |
| `raw/excel/header_probe.xlsx` | 4 sheets, headers on rows 1/2/3/5 | Header detection (§14.1) |
| `raw/excel/budget_input.csv` | 12 months of budget and actual | What-if (§14.9) |

The main workbook has a merged title, a header on row three, a total row inside
the data region, 12 text dates, 40 trailing-space regions, 8 raw category labels
that collapse to 4, a hidden stale sheet, a hard-coded override, a cached total
of zero in `Raw Data!H204`, and a summary cached total of $182,346.72 that
disagrees with the $33,293.53 recomputed from the 200 data rows.

Generated reports are written to `reports/ch14_report.xlsx` and
`reports/ch14_monthly_sheets.xlsx`. `openpyxl` reads formulas and cached values;
`xlsxwriter` writes new workbooks with formats, validation, and charts.

## Chapter 15 business fixtures

Chapter 15 applies the whole toolkit to a fictional retailer, so it ships eight
small domain tables under `raw/business/`. Everything is deterministic:

```bash
python scripts/generate_chapter15_sources.py   # writes the files below
python scripts/chapter15_outputs.py            # prints every output block
```

| Path | Rows | What it is | Used for |
|---|---|---|---|
| `raw/business/biz_monthly.csv` | 288 | 24 months x 3 channels x 4 regions: sessions, orders, revenue, cogs, returns, shipping, margin | Metrics, revenue, finance (§15.1, §15.2, §15.7) |
| `raw/business/biz_products.csv` | 30 | Product performance with Q3/Q4 units and revenue | ABC and Pareto (§15.2) |
| `raw/business/biz_campaigns.csv` | 36 | 3 platforms x 12 months: spend, clicks, platform/first/last/linear conversions | Marketing and attribution (§15.3) |
| `raw/business/biz_customers.csv` | 4,000 | Customer revenue, margin, recency, segment, NPS, tickets | Customer analytics (§15.4) |
| `raw/business/biz_events.csv` | 12,949 | Product usage events with user, event, date, platform | Product analytics (§15.5) |
| `raw/business/biz_cohort_activity.csv` | 288 | 24 cohorts x 12 months of active customers | Cohort retention (§15.4) |
| `raw/business/biz_operations.csv` | 12 | Fulfilment steps: minutes, defects per 1,000, cost per order | Operations (§15.6) |
| `raw/business/biz_finance.csv` | 24 | Budget vs actual, cogs, margin, opex | Finance for analysts (§15.7) |

The tables are independent slices of the same fictional business and reconcile
to a common set of totals: revenue $23,521,127.76, 131,922 orders, 1,237,961
sessions, and 38.2% gross margin. The campaign spend of $1,066,150.46 covers 12
months of 2025 while the monthly table covers 24 months, a deliberate
mixed-window inconsistency that the chapter's metrics-catalog section flags.
Platform-reported conversions (16,317) exceed warehouse last-touch orders
(10,349), leaving a 5,968 gap to reconcile. The customer table is a sample of
4,000 buyers, not the full buying population, so customer revenue does not sum
to total revenue.

Two figures are deliberately not cash: the 15-minute customs saving is worth
$67,500 a month only if saved hours reduce cost or add capacity, and the
accrual margin is not a cash margin. Both caveats are stated where the figures
appear.

## Chapter 16 predictive-analytics fixtures

Chapter 16 builds and validates churn and price models, so it ships three small
tables under `raw/ml/` and nine figures. Everything is deterministic:

```bash
python scripts/generate_chapter16_sources.py   # writes the files below
python scripts/chapter16_outputs.py            # prints every output block
```

| Path | Rows | What it is | Used for |
|---|---|---|---|
| `raw/ml/ml_churn.csv` | 6,000 | Customer snapshots with 11 clean features, a 4.07% churn label, and 3 leaky columns | Supervised framing, models, evaluation (§16.2–§16.9) |
| `raw/ml/ml_listings.csv` | 2,000 | Property listings with a price function containing two nonlinear interactions | Regression baselines and regularisation (§16.3, §16.7) |
| `raw/ml/ml_dup_entities.csv` | 12 | Six entities with conflicting labels across two snapshots each | Duplicate-entity leakage (§16.2) |

The churn label is "cancelled within 90 days of the snapshot" on eight quarterly
snapshots from 2024-01 to 2025-10. There are 244 positives and 5,756 negatives.
The temporal split trains on the first six quarters (4,474 rows, 191 positives)
and tests on the last two (1,526 rows, 53 positives), which is why every
precision figure in the chapter is quoted with those counts.

Three columns are deliberate leaks and must be excluded from any honest model:
`cancellation_flag` is populated only after the event, `legacy_churn_score` is
a prior system's estimate of the same label, and `final_invoice_amount` is
generated after cancellation. A logistic model given all three reaches ROC AUC
1.0000 against 0.7829 for the clean feature set. Group B also pays about $6 more
a month on average than group A, so `monthly_charges` acts as a proxy for the
protected group even when the group column is removed.

Nine figures are written to `images/ch16_*.png`: leakage ROC comparison,
ROC/PR curves, confusion matrix, reliability diagram, cumulative gain,
permutation importance, cost by threshold, bias–variance curves, and quarterly
drift. scikit-learn 1.9 provides the models; `statsmodels` is not required.

## Chapter 17 anomaly fixtures

Chapter 17 finds rare events and then recommends an action, so it ships four
small tables under `raw/anomaly/` and nine figures. Everything is deterministic:

```bash
python scripts/generate_chapter17_sources.py   # writes the files below
python scripts/chapter17_outputs.py            # prints every output block
```

| Path | Rows | What it is | Used for |
|---|---|---|---|
| `raw/anomaly/anom_daily.csv` | 730 | Two years of daily revenue, orders, and sessions with labelled outage, level shift, promo, and spike days | Baselines and time-series detection (§17.2–§17.3) |
| `raw/anomaly/anom_orders.csv` | 12,000 | Orders with six numeric features, a 0.9% fraud rate, and 40 injected labelled cases | Multivariate detection, rules, triage (§17.4–§17.6) |
| `raw/anomaly/anom_budget.csv` | 80 | Four concave channel response curves sampled every $5,000 | Optimisation with `linprog` (§17.7) |
| `raw/anomaly/anom_assumptions.csv` | 6 | Monte Carlo inputs with base, low, and high values | Simulation and scenarios (§17.8) |

The daily series contains 684 normal days, 40 level-shift days, a three-day
outage, two spikes, and one promotion. Weekly seasonality is large: Saturday
and Sunday medians are about $27,600 against about $20,100 on Wednesday, so
detectors work on weekday residuals rather than raw revenue. The residual MAD
roughly doubles between the two years, which is a deliberate drift signal.

The order table has 108 frauds and 40 injected extreme cases with `known_case =
1`. Those injections are for a smoke test only; they were built from the same
velocity, distance, and amount rules used to detect them, so precision measured
on them is not operational precision. `ruptures` and `PuLP` are not installed;
change points use rolling comparisons and the optimisation uses
`scipy.optimize.linprog` with ten linear segments per channel.

Nine figures are written to `images/ch17_*.png`: seasonal residual, control
charts, level shift, multivariate score distributions, precision at k, response
curves, Monte Carlo histogram, tornado chart, and decision options.

## Chapter 18 reporting fixtures

Chapter 18 is about communicating findings and handing work over, so it ships
five small tables under `raw/reporting/` and eight figures. Everything is
deterministic:

```bash
python scripts/generate_chapter18_sources.py   # writes the files below
python scripts/chapter18_outputs.py            # prints every output block
```

| Path | Rows | What it is | Used for |
|---|---|---|---|
| `raw/reporting/rep_findings.csv` | 30 | Findings from a mock 30-page analysis, each with confidence, relevance, and audience | Audience analysis and finding-first structure (§18.1–§18.2) |
| `raw/reporting/rep_kpi.csv` | 12 | KPIs with actual, target, prior, direction, and owner | RAG reports and variance (§18.3) |
| `raw/reporting/rep_scenarios.csv` | 4 | Low, base, and high values for four assumptions | Communicating uncertainty (§18.6) |
| `raw/reporting/rep_questions.csv` | 5 | The five questions stakeholders actually ask | Audience and recommendation sections |
| `raw/reporting/rep_audiences.csv` | 4 | Executive, manager, technical, and client profiles | Audience analysis (§18.1) |

Only four findings have relevance ten and four are tagged for executives, so a
one-page brief is a selection from a pool mostly written for managers. RAG
status is defined as green at target, amber within 7% in the unfavourable
direction, and red beyond that; NPS uses a 0.10-point amber band because a ratio
of two negative numbers is meaningless.

Eight figures are written to `images/ch18_*.png`: a bad and a good chart for the
same data, a RAG panel, a rendered one-page brief, a ten-slide storyboard, an
uncertainty range plot, an attention chart, and a data-lineage diagram. Jinja2
provides templating; `papermill` and Streamlit are not installed and are shown
as guarded examples with plain-function and static-page equivalents.

## Chapter 19 project data

Chapter 19 is the first full project: a two-quarter retail sales analysis. It
ships a dedicated SQLite database under `data/raw/project19/` and eight figures:

```bash
python scripts/generate_chapter19_sources.py   # rebuilds the database and figures
python scripts/chapter19_outputs.py            # prints every output block
```

| Path | Rows | What it is | Used for |
|---|---|---|---|
| `data/raw/project19/kestrel_project.db` | — | Six tables: orders 25,267, order lines 63,091, products 40, customers 8,000, stores 12, returns 1,562 | All seven phases |
| `data/raw/project19/marketing_spend.csv` | 24 | Monthly spend by four channels | Promotion analysis |
| `data/raw/project19/finance_revenue.csv` | 2 | Finance's reported net revenue by quarter | Reconciliation |

The comparison is Q1 2026 against Q1 2025. Q1 2025 is a 12-week extract and Q1
2026 a 13-week extract, so a raw comparison is invalid; the project uses a
matched 12-week window as a sensitivity check. Planted defects are 60 duplicate
order keys ending in `-D`, 120 missing unit costs, 773 partial returns, and a
$12,450 post-close finance adjustment.

Net revenue falls 6.25% from $4,385,247.90 to $4,111,047.00. The
price–volume–mix bridge closes exactly: volume +$225,450.54, price −$486,563.77,
mix −$13,087.66. South and Midwest account for 79.9% of the decline. The
recommendation caps discounts at 15% on the top-20 products, owned by the
pricing lead at $60k cost against $420k expected value.

Eight figures are written to `images/ch19_*.png`: monthly trend, bridge,
regional change, discount–margin curve, promotions, returns, product
concentration, and the four-chart executive pack.

## Chapter 20 churn project data

Chapter 20 is the second full project: a membership churn and retention
analysis. It ships a SQLite database and four CSV files under
`data/raw/project20/`, plus eight figures:

```bash
python scripts/generate_chapter20_sources.py   # rebuilds data and figures
python scripts/chapter20_outputs.py            # prints every output block
```

| Path | Rows | What it is | Used for |
|---|---|---|---|
| `data/raw/project20/kestrel_membership.db` | — | Five tables: members 12,000, orders 114,110, tickets 11,406, sessions 349,174, email 202,237 | All seven phases |
| `data/raw/project20/fixed_horizon.csv` | 16,276 | Two snapshots with 90-day labels, 11 features, and 3 leaky columns | Preparation, modelling, validation |
| `data/raw/project20/cohort_activity.csv` | 468 | 36 signup cohorts x 13 tenure months | Retention triangle |
| `data/raw/project20/exit_survey.csv` | 400 | Themed exit responses | Survey themes |
| `data/raw/project20/finance_margin.csv` | 3 | Margin per month and observed churn by plan | CLV and payback |

The contractual churn rate over three years is 13.54%, but the 90-day rate is
about 2.5%; the project uses one definition throughout. Three columns
(`cancellation_reason`, `days_after_end`, `final_refund`) are post-churn leaks:
adding the encoded cancellation reason takes ROC AUC from 0.6195 to 0.9895.
86.5% of members are censored at the cutoff, which is why a fixed-horizon label
is used.

Eight figures are written to `images/ch20_*.png`: retention triangle, survival
curve, tenure, segments, engagement, survey themes, calendar effect, and CLV.

## Chapter 21 marketing project data

Chapter 21 is the third full project: marketing reconciliation, budget
allocation, and a checkout A/B test. It ships a SQLite database and two CSV
files under `data/raw/project21/`, plus eight figures:

```bash
python scripts/generate_chapter21_sources.py   # rebuilds data and figures
python scripts/chapter21_outputs.py            # prints every output block
```

| Path | Rows | What it is | Used for |
|---|---|---|---|
| `data/raw/project21/kestrel_marketing.db` | — | Six tables: ad_daily 1,092, platform_daily 1,092, orders 20,820, touches 34,882, campaigns 6, experiment 12,000 | All seven phases |
| `data/raw/project21/finance_assumptions.csv` | 9 | AOV, margin, FX, annual users, MDE, alpha, and target power | CAC, payback, power |
| `data/raw/project21/tracking_incidents.csv` | 6 | Duplicate, outage, currency, and timezone defects | Reconciliation |

Spend rose 18.0% from Q4 2025 to Q1 2026 while revenue rose 0.6%. Platform
conversions over-credit warehouse orders by 2.2% on Search, 20.6% on Social,
and 49.9% on Display. The incident log documents a duplicate pixel on
2026-02-14, a three-day Search outage, a euro-billed Display campaign, and a
16% share of orders crossing UTC midnight. Last-click, first-click, and linear
attribution give different channel ROAS but the same ranking; summing the three
models would triple-count $3.27M of paid revenue.

The checkout experiment enrolled 12,000 users (6,000 per arm): conversion rose
1.25 points (p = 0.0405) but the refund guardrail rose 0.70 points (p = 0.034),
and no segment survives Benjamini-Hochberg. The decision is to hold. The
budget recommendation shifts $48k a quarter from Display to Search and Social,
modelled to add $87k-$101k across conservative and optimistic response curves.

Eight figures are written to `images/ch21_*.png`: reconciliation gap, spend
trend, platform versus warehouse ROAS, saturation curves, experiment
conversion, refund guardrail, attribution models, and the dashboard.

## Snapshot

| Property | Value |
|---|---|
| Grain | One order line; an order may have several lines |
| Columns | 14 |
| Distinct orders | 213,040 |
| Distinct order lines (after de-duplication) | 487,936 |
| Months covered | 27 (2024-01 to 2026-03) |
| Date range | 2024-01-01 to 2026-03-31 |
| Average lines per order | ~2.3 |
| Extraction date (in narrative) | 2026-04-02 |

## Data dictionary

| Column | Type | Meaning | Allowed values / notes |
|---|---|---|---|
| `order_id` | text | Order key | `O-######`; repeats across the lines of one order |
| `order_date` | text | Date the order was placed | `YYYY-MM-DD` in the normal case; 14 rows hold invalid tokens by design |
| `customer_id` | text | Customer key | `C-######`; blank for guest checkouts |
| `channel` | text | Sales channel | `Web`, `App`, `Phone` |
| `region` | text | U.S. region | `Midwest`, `Northeast`, `South`, `West`; blank ~1.8% of rows |
| `category` | text | Product category | `Camping`, `Footwear`, `Apparel`, `Accessories` |
| `product_id` | text | Product key | `P-####` |
| `quantity` | integer | Units on the line | 1–8; capped at 1 for items over $250 |
| `unit_price` | float | Price per unit before discount | List price with small variation |
| `discount_pct` | float | Discount percent | 0–70; blank means no discount was recorded (treated as 0 in revenue) |
| `shipping_cost` | float | Shipping charged on the line | 0 (free shipping or returned order), 3.95, 7.95, 9.95, 12.95, or a rare 49.95 express rate |
| `returned` | integer | Return flag | `0` or `1` |
| `revenue` | float | Revenue after discount, before returns and shipping | `quantity * unit_price * (1 - discount_pct / 100)` |
| `store_id` | text | Originating store | `S-01` to `S-12` |

Derived in the chapter:

* `return_value = revenue * returned`
* `net_revenue = revenue - return_value - shipping_cost`

## Known defects (deliberate)

| Defect | Count | Why it exists |
|---|---|---|
| Exact duplicate rows | 1,184 | Simulated March 2026 migration export |
| Unparseable `order_date` values | 14 | Dates stored as text, with real-world junk |
| Missing `region` | 8,893 | Incomplete source field |
| Missing `customer_id` | 2,189 | Guest checkouts |
| Blank `discount_pct` | 12,471 | No discount recorded, not unknown |

## Reproducing the file

The data is generated deterministically from a fixed seed, so the same command
always produces the same bytes.

```bash
python scripts/generate_kestrel_orders.py
```

SHA-256 of the committed snapshot:

```
31224C0149A1AE54A492618E949E40C6D45B5C54108B66D197A3EEC0041FED8B
```

To print the checks used in Chapter 1 (shape, grain, duplicates, Q1 totals,
AOV, channel/category splits, returns, and the sensitivity table):

```bash
python scripts/verify_kestrel_orders.py
```

Requirements: Python 3.12, `numpy`, `pandas`.

## Key figures used in Chapter 1

| Metric | Q1 2025 | Q1 2026 |
|---|---|---|
| Net revenue | $4,392,118 | $4,119,998 |
| Orders | 23,641 | 22,890 |
| Return rate (lines) | 7.3% | 8.1% |
| Average order value | $185.78 | $179.99 |

Year-over-year change in net revenue: **-6.2%**.

## Licensing and GitHub

The dataset is synthetic and contains no real people or companies. Treat it as
public-domain and free to redistribute with the book's companion code.

`kestrel_orders.csv` is about 41 MB. That is under GitHub's 100 MB hard limit
but above its 50 MB warning, so pick one of these when you publish:

1. **Git LFS (recommended for this file).**
   ```bash
   git lfs install
   git lfs track "data/raw/*.csv"
   git add .gitattributes
   ```
2. **Commit directly.** It will work, but every clone downloads the full file
   and GitHub will warn.
3. **Ship the generator only.** Commit `scripts/generate_kestrel_orders.py`
   and add `data/raw/*.csv` to `.gitignore`; readers regenerate the file in
   seconds.

If you only publish a sample, keep enough rows to preserve the Q1 2025/Q1 2026
comparison; the monthly aggregates in Chapter 1 depend on the full 27 months.
