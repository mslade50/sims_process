# Design system v2 (golf desk)

Direction: an expert golf desk with the clarity of a scorecard and the precision of a trading blotter.
Source of truth: `app/tokens.css` (tokens), `app/design.css` (global restyle + primitives CSS), `app/shell.css` (shell),
`app/ui.tsx` (primitives; each is documented in its header comment), `app/ShellParts.tsx` (event switcher, tab bar).
Everything a page needs is already global: restyled `.panel`, `.kpi`, tables, `.segmented`, `.empty-state`, `.page-intro`.

## Tokens (never use raw colours or sizes)
- Surfaces: `--background` (ground), `--surface` (data surface), `--surface-raised`, `--surface-soft`, `--popover`. Rules: `--line`, `--line-strong`. Hover tints: `--tint-1`, `--tint-2`.
- Text: `--foreground`, `--muted-strong`, `--muted` (all >= 4.5:1 on every surface, both themes; `tests/theme.test.mjs` enforces it).
- Accent = selection only: `--accent`, `--accent-soft`, `--accent-ink`. User can switch accent; never hard-code blue.
- Semantic: `--pos` (favourable), `--neg` (adverse), `--warn` (actionable uncertainty), plus `-soft` fills. Green/red only for good/bad, never decoration.
- Model vs book: `--model`, `--market`; gap rail: `--rail-model/-market/-pos/-neg/-zero/-track`.
- Space (4px): `--sp-1..--sp-16` = 4 8 12 16 20 24 32 40 48 64. Bar heights: `--bar-h` 56, `--tabbar-h` 60.
- Radius: `--radius-control` 8 (buttons, inputs, chips), `--radius-lg` 12 (panels, popovers). Nothing else.
- Type (Inter Var, self-hosted `public/fonts/inter-var.woff2`, preloaded): `--fs-h1` 28 (24 phone), `--fs-h2` 18, `--fs-h3` 15, `--fs-body` 14 desktop / 16 phone, `--fs-data` 14, `--fs-small`/`--fs-label` 12 (the minimum), `--fs-kpi` 24-30.
  Numbers: tabular figures are on for tables, `.kpi`, `.stat`, `.delta`, `.num`. Put `className="num"` on any other compared number.
- Motion: `--dur-fast` 120ms, `--dur` 200ms, `--ease`. Animate transform/opacity only. Reduced motion is handled globally.

## Shell (do not rebuild per page)
- Desktop: 216px sidebar (Bet card pinned first for the owner, grouped nav, active = accent tint + left bar), 56px context bar:
  `page / [PGA  Event name  course v] [stage chip] [freshness] ... search, theme`. The switcher sets `?e=<event_uid>` and reloads.
  Stage = newest saved run ("Pre-event", "Live, after R2", "Final"). Freshness colour: green fresh, amber aging, red stale.
- Phone (<= 820px): two-row context (switcher + search + theme / stage + freshness), bottom tab bar (This week, Players, Bet card or Why priced for non-owners, Weather, More). More opens the full grouped nav as a bottom sheet. Content has bottom padding for the bar and safe-area insets.
- Pages must not render their own event pickers that duplicate the bar unless they compare events side by side.

## Page layout recipe
```tsx
<PageHeader eyebrow="Week" title="Weather" purpose="How the forecast moves scoring and who it helps." actions={<SegmentedControl .../>} />
<OnThisPage sections={[{ id: "field", label: "Field" }, { id: "rounds", label: "Rounds" }]} />   // only for pages > ~2 screens
<div className="kpi-grid"><Stat label="Field effect" value="-0.59" unit="strokes" delta={-0.12} inverse note="vs last edition" /></div>
<Section id="field" title="Field difficulty" purpose="Whole-field strokes per round."> ...table or chart... </Section>
```
Order: answer first (header + 3-4 Stats or the single decisive row), then sections by decreasing importance, methodology last behind a `<details>`.
Use `Panel` only for a self-contained tool (filters, forms, charts with controls). Sections, rules and compact rows are the default.

## Primitives (all in `app/ui.tsx`, re-exported from `components.tsx`)
- `PageHeader`, `Section`, `OnThisPage`: above. `PageIntro` still works and renders the same header (accepts `actions`).
- `Stat` (alias look of `Kpi`): value pre-formatted, `unit` beside it, `delta` always signed with arrow; `inverse` when lower is better.
- `Delta`: never colour only (arrow + sign + sr-only word).
- `GapRail`: `<GapRail model={24.0} book={19.7} />` renders the rail and "model 24.0% vs book 19.7% = +4.3 pts". Use in price rows (player, model, book, gap rail, driver).
- `Chip tone="accent|positive|negative|warning|neutral"`: stage/status/category; text carries the meaning. `Badge` is the older live-dot pill.
- `SegmentedControl`: aria-pressed buttons; use for 2-6 exclusive options. Raw `.segmented` markup should add `aria-pressed`.
- `EmptyState title detail nextCheck`: one line each: what is missing, why, when it is checked again ("Next check: Thursday 6:00 ET"). Sentence case, no giant boxes, never blank.
- `Skeleton`, `SkeletonPage`: exact-height placeholders; no spinners.
- `InfoTip definition="..."`: dotted jargon term, click opens a one-sentence definition (use for CLV, edge, cut line ...).
- `AnimatedNumber` / `data-flash`: value-change highlight once.
- `DataTable`: numeric columns right-align automatically (`num`), sticky header, compact rows; pass `mobileColumns` for phone.

## Do
- Lead each page with the answer; one h1; units next to values; consistent decimals per column; signed deltas (+4.3, -0.9).
- Tables over cards for comparisons; right-align and tabulate numbers; name the unit in the header.
- Colour semantics: green favourable, red adverse, amber uncertain, blue selected. Pair colour with a sign, arrow or word.
- 44px targets on phone (`min-height: 44px`), 16px inputs on phone, visible focus (global `:focus-visible` ring; do not remove outlines).
- Test at 360px and in both themes; run `node scripts/visual-check.mjs --routes <page>`.

## Don't
- No nested bordered cards (`.panel .panel` is flattened on purpose), no dashed empty boxes, no gradients or glow, no hover lift.
- No concatenated titles ("A · B"), no uppercase except short structural labels (`.eyebrow`, table headers, nav group labels).
- No text under 12px, no raw hex, no new radii, no `transition: all`, no per-page fonts, no new heavy dependencies.
- Don't hide analytical depth to look clean: demote it to a Section, a table or a `<details>`, one action away.

## Known gaps for page agents
Many page CSS files still hard-code `font-size: 11px` and 16px radii: replace with `--fs-small` / `--radius-lg` when you touch them.
Several pages render their own event pickers (This week, Weather, Players); fold them into the bar's event where possible.
Tap targets under 44px remain on page-level controls (Bet card filters, Run checkboxes, weather sliders, pagination).
