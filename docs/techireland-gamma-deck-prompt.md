# Gamma prompt — TechIreland pitch deck (7 minutes, with live demo)

Seven cards, because the demo takes roughly two and a half minutes of the seven.
Paste everything between the rules into Gamma (Create > Paste in text > "Generate
from this outline"). Set cards to 7 and choose "Preserve my text" so Gamma does not
rewrite the bullets.

---

Create a 7-card judge pitch deck for a 7-minute spoken pitch that includes a live
software demo. This is a technical product pitch to judges at an Irish tech
competition, not a marketing brochure. The deck is deliberately short: the slides
frame the demo, they do not duplicate it.

RULES — follow exactly:
- Maximum 5 bullets per card. Maximum 12 words per bullet. No paragraphs of body
  text anywhere. The bullets are speaker cues, not the script.
- Do NOT invent statistics, percentages, market sizes, funding figures or customer
  numbers. If a number is not written below, do not add one.
- No emoji, no clipart, no cartoon fish, no stock "business handshake" imagery.
  Imagery should be underwater footage, camera hardware, Irish coastline and
  rivers, or clean data-console screenshots.
- Put the longer explanation in the presenter notes of each card, not on the card.

VISUAL THEME — match this exactly:
- Palette: deep forest green #1E3A27 (primary), mid forest #4A6B53, sage #9DB2A3,
  near-black #0B140E for full-bleed footage cards, off-white #F5F6F4 for content
  cards, ink #202622 for body text, steel blue #2F6690 for links and highlights,
  amber #A66A24 used only for a "needs review" or warning accent.
- Title and closing cards: dark green gradient, 168deg, #35533D to #26422F to
  #16281C, white text.
- Content cards: off-white #F5F6F4 ground, dark ink text, a single thin hairline
  rule #D7DCD8 as the only divider. No drop shadows, no rounded "bubble" cards.
- Type: Inter (or system sans) for everything; a monospace face for any number,
  species name or metric. Headings sentence case, tight, no exclamation marks.
- Overall feel: an operator console / scientific instrument. Restrained,
  high-contrast, no gradients other than the green one above.

Here are the 7 cards.

---

CARD 1 — Title
Fish Monitor
Sub-line: Underwater video into fish data, automatically.
Footer: TechIreland App | [Your name] | [Team name]
Visual: full-bleed dark underwater footage, green-black grade.

---

CARD 2 — "The footage is the bottleneck"
- Irish waters are already monitored by camera, not by eye
- Rivers, fish passes, farm pens, offshore sites — all recording
- Someone still has to watch every hour of it
- So agencies sample, they do not count
- Inland Fisheries, the Marine Institute, aquaculture, offshore wind —
  all reporting to EU directive deadlines

---

CARD 3 — "Live"
One line only, large: Live demo.
- Upload, detect, track, review, export
Visual: near-black #0B140E ground, no other content. This card stays on screen
while the presenter switches to the application.
Presenter note: 2.5 minutes. Show live camera tracking on Galway Bay footage, then
the review queue with a flagged low-confidence track, then the CSV export. Do not
narrate the architecture — show the operator's day.

---

CARD 4 — "Why it works this way"
- Built around the reviewer, not the model
- Every track keeps model confidence plus a human decision
- Low confidence is flagged "review required", never guessed
- Open detection core, a paid classifier only on the best crops
- Runs on-premise in Docker — footage never leaves the institution
Presenter note: AI is doing the one job a human cannot — watching 24/7 at frame
rate. VIAME does detection and tracking; quality gates decide which crops are worth
paying to identify; multi-frame consensus and an Irish-waters regional filter reject
implausible labels; a human signs off. Capped API spend, no runaway inference bill.

---

CARD 5 — "Good for Irish waters"
- Non-invasive: no electrofishing, netting or handling mortality
- Continuous counts instead of sampled snapshots
- Evidence that fish passes and barriers actually work
- Early warning for invasive species and range shifts
- Unlocks video archives already paid for and sitting idle

---

CARD 6 — "Who pays"
- State agencies: monitoring capacity without more headcount
- Aquaculture: daily pen data, tied directly to stock value
- Offshore wind and hydro: EIA baseline and consent evidence
- Model: paid pilot, then per-site annual licence, on-premise option
- Innovasea proves fish data is a real, funded, profitable market
Presenter note: Innovasea built a global business on fish tracking and aquaculture
data. Buyers already budget for monitoring; what they lack is throughput. We sell
the missing analysis layer, not a new obligation.

---

CARD 7 — "What we learned, what's next"
- Spoke to SmartBay's senior developers and a marine biologist,
  Marine Institute Galway — both IMAGINE project veterans
- Learned: Irish species identification is the real gap
- Learned: labelled Irish footage is scarce, turbidity is the hard part
- Next: species model trained on Irish fish, using SmartBay footage we have access to
- Then one site pilot, then the same system anywhere
Footer: [Your name] | [email] | [repo or demo link]
Visual: dark green gradient, same as Card 1.

---

## Notes before you paste

- **Timing at 7 minutes:** title 15s, card 2 55s, demo 150s, card 4 50s, card 5 35s,
  card 6 45s, card 7 50s — about 6:40 with a little slack. If the demo overruns, cut
  card 5 to two bullets. Never cut card 2 or card 7.
- **Rehearse the demo cold-started.** Two and a half minutes disappears fast if a
  container is still booting. Have the app already running and the review queue
  already populated before you stand up.
- **Card 7 names real people and a real dataset.** Confirm with your Marine
  Institute contacts that they are happy to be cited, and use roles rather than
  names if they would rather not appear on a deck the judges keep.
- **Card 6 carries no Innovasea figure on purpose.** If you want one, look it up and
  type it in yourself — Gamma will fabricate one if you leave a gap.
- **Have a fallback for the demo:** a 60-second screen recording on card 3 in case
  the venue network or the camera feed fails. Judges forgive a recording; they do
  not forgive three minutes of a spinner.
