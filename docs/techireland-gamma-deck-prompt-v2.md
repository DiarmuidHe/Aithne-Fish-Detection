# Gamma prompt v2 — Aithne pitch deck (7 minutes, with live demo)

Rewritten after seeing the v1 render. Fixes the false "zero automated counts" claim,
pulls the theme back off Gamma's navy-and-teal default, kills the dead space on
cards 3, 5 and 7, and stops Gamma inventing its own pull-quotes.

Two ways to use this:
- **Regenerate** (recommended): new Gamma deck, Create > Paste in text, paste
  everything between the rules, 7 cards, "Preserve my text".
- **Patch the existing deck**: skip to "Per-card fixes for the deck you already
  have" at the bottom and apply them card by card.

---

Create a 7-card judge pitch deck for a 7-minute spoken pitch that includes a live
software demo. Technical product pitch to judges at an Irish tech competition. The
product is called **Aithne**. The deck is deliberately short: the slides frame the
demo, they never duplicate it.

TEXT RULES — follow exactly:
- Use my bullet text verbatim. Do not rewrite, expand or re-order it.
- **Do not invent pull-quotes, callouts, slogans, statistics, percentages, market
  sizes, funding figures or customer numbers.** The only callout text permitted is
  text I have explicitly marked CALLOUT below. If a card has no CALLOUT line, it
  gets no callout.
- Every bullet must fit on one line, two at absolute most. Never three.
- No paragraphs of body text. Bullets are speaker cues, not the script.
- Put longer explanation in presenter notes, never on the card.

LAYOUT RULES — follow exactly:
- Left-align every heading and every body text block. The only centred card is
  card 1.
- No dead quadrants. If a card has 4 items, grid them 2×2 at equal size. If it has
  5, use one horizontal row. Never 3-then-2 with an empty corner.
- No stray arrow glyphs, navigation chevrons, page arrows or decorative icons
  floating outside a content block.
- Inline code styling only on the exact phrase `review required` and on card 3's
  footer line. Nowhere else.

VISUAL THEME — this is a green-black deck. Match it exactly:
- **Never use navy, blue-grey, slate or teal.** Gamma's default dark theme is
  wrong for this brand.
- Dark grounds: green-black #0B140E, or the gradient #35533D to #26422F to #16281C
  at 168deg. Every dark card uses one of those two.
- Accent on dark: sage #9DB2A3. Panel fills on dark: forest #4A6B53 or #1E3A27.
- Light content cards (if any): off-white #F5F6F4 ground, ink #202622 text,
  hairline #D7DCD8 as the only divider.
- Steel blue #2F6690 for links only. Amber #A66A24 only where the words
  `review required` or a warning appear. Nothing else gets a colour.
- No drop shadows, no rounded bubble cards, no glow.
- Type: Inter or system sans throughout; monospace for any number, species name or
  metric. Headings sentence case, tight, no exclamation marks.
- Feel: an operator console / scientific instrument. Restrained and high-contrast.

Here are the 7 cards.

---

CARD 1 — Title. Centred, full-bleed dark underwater footage graded green-black.
Aithne
Sub-line: Underwater video into fish data, automatically.
Footer line, small, sage #9DB2A3: TechIreland App | [Your name] | [Team name]
Layout note: put the lockup in the vertical centre, not the lower third. The card
must not read as half-empty.

---

CARD 2 — "The footage is the bottleneck"
Layout: two columns. Bullets left. One forest-green #1E3A27 panel right holding the
callout only.
- Irish waters are monitored by camera, not by eye
- Rivers, fish passes, farm pens, offshore sites
- Someone still has to watch all of it
- So agencies sample instead of counting
- Everyone here reports to EU directive deadlines
CALLOUT (use this text exactly, nothing else):
Recorded everywhere. Counted almost nowhere.
Presenter note: automated counters do exist — river counters, sonar, VIAME itself.
What does not exist is a way for an operator to point at an archive and get
reviewed, signed-off numbers out. The gap is throughput and workflow, not detection.

---

CARD 3 — "Live demo"
Layout: near-black #0B140E, almost empty by design. Heading upper left. The five
steps as ONE horizontal row across the lower third, small equal-width labels, no
chevrons and no numbered blocks. Nothing in the middle of the card.
Steps: Upload / Detect / Track / Review / Export
Footer, monospace, small, sage: demo running live — see screen
Presenter note: 2.5 minutes. Live camera tracking on Galway Bay footage, then the
review queue with a flagged low-confidence track, then the CSV export. Show the
operator's day; do not narrate the architecture.

---

CARD 4 — "Why it works this way"
Layout: exactly four equal tiles in a 2×2 grid, each a short bold headline plus one
line under it. Then one full-width line beneath the grid.
- Reviewer-first — every track keeps model confidence and a human decision
- Never guesses — low confidence is flagged `review required`
- Costed AI — the paid classifier runs only on the best crops
- Stays put — on-premise in Docker, footage never leaves the institution
Full-width line under the grid: Open detection core. No per-frame vendor toll.
Presenter note: AI does the one job a human cannot — watching 24/7 at frame rate.
VIAME handles detection and tracking; quality gates decide which crops are worth
paying to identify; multi-frame consensus and an Irish-waters regional filter reject
implausible labels; a human signs off. API spend is capped by design.

---

CARD 5 — "Good for Irish waters"
Layout: exactly four equal tiles in a 2×2 grid, small line icon, bold headline, one
line under it. Then one full-width line beneath the grid. Tight vertical spacing —
no empty lower half.
- Non-invasive — no electrofishing, netting or handling mortality
- Continuous counts — not sampled snapshots
- Barrier evidence — proof that fish passes actually work
- Early warning — invasive species and range shifts
Full-width line under the grid: And it unlocks video archives already paid for.

---

CARD 6 — "Who pays"
Layout: keep this one as it is — three buyer tiles left, numbered revenue model
right, one highlighted strip across the bottom. Change the highlight strip from teal
to forest green #1E3A27 with sage text.
Buyer tiles:
- State agencies — monitoring capacity without more headcount
- Aquaculture — daily pen data, tied directly to stock value
- Offshore wind and hydro — EIA baseline and consent evidence
Revenue model, numbered 01 / 02 / 03:
- Paid pilot
- Per-site annual licence
- On-premise option
CALLOUT strip (use this text exactly):
Innovasea proves fish data is a real, funded, profitable market.

---

CARD 7 — "What we learned, what's next"
Layout: two clean columns, both left-aligned, equal width, no timeline graphic and
no stray arrows. Left column labelled LEARNED, right column labelled NEXT on a
forest #1E3A27 panel. Left-align the text inside both columns — nothing centred.
LEARNED:
- Spoke to SmartBay's senior developers, Marine Institute Galway
- And a marine biologist there — both IMAGINE project veterans
- Irish species identification is the real gap
- Labelled Irish footage is scarce; turbidity is the hard part
NEXT:
- A species model trained on Irish fish
- Using SmartBay footage we already have access to
- Then one site pilot
- Then the same system, anywhere
Footer, small, sage: [Your name] | [email] | [repo or demo link]

---

## Per-card fixes for the deck you already have

If you would rather patch than regenerate, these are the seven changes that matter,
in priority order:

1. **Card 2, the callout is wrong and must change.** "Zero automated counts" is
   false — river counters, sonar counters and VIAME all exist, and a Marine
   Institute judge will know it. It also contradicts card 4, where you credit an
   open detection core. Replace with: *Recorded everywhere. Counted almost
   nowhere.*
2. **Theme is navy and teal; the brand is green-black and sage.** Set the dark
   ground to #0B140E or the #35533D→#16281C gradient, and swap every teal accent
   and teal strip for sage #9DB2A3 or forest #1E3A27.
3. **Card 3 is 70% empty.** Drop the vertical numbered chevrons and lay
   Upload/Detect/Track/Review/Export as one small horizontal row along the bottom.
4. **Card 4 grids 3-then-2 with a hole.** Cut to four tiles in 2×2 and move "open
   detection core" to a full-width line underneath.
5. **Card 5 has an empty lower half.** Same fix: four tiles 2×2, "unlocks archives"
   becomes the line beneath.
6. **Card 7 is the messiest card and it is your closing card.** Remove the stray
   back-arrow, left-align the text in both columns, drop the timeline dots, and add
   your name, email and demo link in the footer.
7. **Card 1 has no footer.** Add name, team and TechIreland App, and lift the
   lockup to the vertical centre.

Still outstanding regardless of which route you take:

- **Confirm your Marine Institute contacts are happy to be named** on card 7, or
  cite them by role only. The deck stays with the judges.
- **Card 6 now carries pricing.** The three prices, the tender-threshold line and a
  sources appendix are in `docs/techireland-gamma-deck-addendum.md` as two paste
  blocks. Type any figure in yourself; Gamma will fabricate one given the chance.
- **Timing:** 15s title, 55s card 2, 150s demo, 50s card 4, 35s card 5, 45s card 6,
  50s card 7 ≈ 6:40. If the demo overruns, cut card 5 to two tiles. Never cut card 2
  or card 7.
- **Demo insurance:** app already warm with a populated review queue before you
  stand up, and a 60-second screen recording behind card 3 in case the feed drops.
