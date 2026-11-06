# UI overhaul rubric

Target: every category scores 9.5/10 or higher from BOTH Claude and Codex (gpt-6.1-sol).
Benchmark bar ("spectacular"): Linear app, Stripe Dashboard, Vercel dashboard, Data Golf,
The Athletic / FiveThirtyEight interactives, Apple Sports, Kalshi/Robinhood market pages.
A 10 means clearly better than the best of those for this job (a private golf pricing and
betting dashboard used on phone and desktop by one expert owner).

Score each category 0-10 (one decimal). For anything under 9.5 give the specific
page/viewport/theme and the concrete change that would lift it.

| # | Category | What a 9.5+ looks like |
|---|----------|------------------------|
| 1 | Information architecture & wayfinding | Owner always knows where they are, where everything lives and how to get to it in 1-2 taps; nav labels match mental model; current page, event and data age obvious. |
| 2 | Clarity of meaning | Every number, chip, colour and chart says what it is, its unit, direction (good/bad) and why it matters; jargon explained in place; no guessing. |
| 3 | Visual hierarchy & layout | The one thing that matters most on each page is seen first; scannable sections; generous yet dense rhythm; no clutter or dead space. |
| 4 | Typography | Clear type scale, tabular numerals for data, comfortable line length, strong headings, legible small text. |
| 5 | Colour, theming & contrast | Cohesive palette with semantic meaning, beautiful in light and dark, WCAG AA contrast, colour never the only signal. |
| 6 | Data visualisation | Charts pick the right form, are labelled directly, readable on phone, consistent, accessible, and answer a question. |
| 7 | Interaction & feedback | Hover/focus/active/loading/empty/error states, transitions, instant response, forms and actions confirm clearly. |
| 8 | Responsiveness & mobile | Phone first-class: no horizontal scroll, 44px tap targets, sensible stacking, tables usable, not endless scroll. |
| 9 | Accessibility | Semantic landmarks, keyboard nav, focus rings, labels, reduced-motion respect, screen-reader names. |
| 10 | Consistency & design system | Same components/patterns/spacing/tokens everywhere; no one-off page styles fighting each other. |
| 11 | Performance & perceived speed | Fast first paint, skeletons not spinners, no layout shift, no console errors. |
| 12 | Craft & delight | Polish, micro-detail, motion with purpose, a distinct identity; feels premium and alive. |

Overall = minimum category score (goal is met only when the minimum is >= 9.5 for both raters).
Scores are logged in docs/ui-overhaul/SCORES.md per round.
