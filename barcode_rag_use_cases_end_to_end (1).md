# Dynamic RAG for Barcode Recovery — Full End-to-End Flow, All Use Cases

Every use case below follows the same 9 steps. No step is skipped in any of them.

**The 9 steps:** (0) what the barcode stores → (1) offline: build the knowledge base → (2) scan + detect → (3) neural decode, top-k guesses → (4) build retrieval query → (5) hybrid retrieval + RRF fusion → (6) verify → (7) mismatch → repair loop → (8) return to user → (9) write-back.

---

## USE CASE 1 — Warehouse worker receiving a shipment (torn / scratched labels)

**Step 0 — What the barcode stores.**
A 13-digit EAN-13 number, e.g. `8901030865278`. Digits 1–3 = country/company prefix, middle digits = product ID, last digit = check digit (computed by a formula from the first 12 — lets us mathematically test a decode). No product name or price inside — those live in the database; the number is just the key.

**Step 1 — Offline: build the knowledge base.**
The truck's manifest (digital packing list) arrives before the truck: "PO#4521, dock 7, 200 items." For each item we create one record:
```
payload: 8901030865278 | product: Dove Soap 100g
PO: 4521 | dock: 7 | category: personal care
visual embedding: vector from a clean photo of this product's barcode/packaging
```
Indexed three ways on the device: SQLite (filter by PO/dock/date), BM25/FTS (fuzzy text search over payload digits and names), HNSW vector index (visual embeddings). When a new truck arrives, a different set of records becomes "active" — the dynamic part.

**Step 2 — Scan + detect.**
Worker points the phone at a box with a torn label. RTMDet finds the barcode in the frame, crops it, corrects tilt.

**Step 3 — Neural decode (top-k).**
The ViT encoder reads the crop into visual features; the character decoder writes the digits one by one. The tear makes digit 12 ambiguous, so beam search keeps multiple guesses:
```
A: 8901030865218  conf 0.61  check digit ✗
B: 8901030865278  conf 0.58  check digit ✓
C: 8901030865216  conf 0.44  check digit ✗
```
Note: the top-confidence guess A is wrong. A normal scanner would fail here and force manual typing.

**Step 4 — Build the retrieval query.**
Query = the guessed strings (A, B, C) + context signals (dock 7, PO#4521 active, morning shift, last 5 scans were personal-care items) + the embedding of the damaged crop itself.

**Step 5 — Hybrid retrieval + RRF.**
Three searches in parallel over the active 200 records:
- BM25: which stored payloads share the most digits with the guesses? → `...5278` matches 12/13 → strong hit.
- Vector search: damaged-crop embedding vs stored clean embeddings → Dove record is nearest.
- SQL: keep only PO#4521 items.
RRF (k=60) merges the ranked lists; a record ranked high in several lists wins:
```
Top retrieved: 8901030865278 | Dove Soap 100g | fused score 0.92
```

**Step 6 — Verify.**
Two checks per guess: (a) math — check digit valid? (b) world — is the payload in the retrieved active set?
Guess B passes both (checksum ✓, on this truck ✓). Guesses A and C fail. → Accept B.

**Step 7 — Mismatch → repair loop (runs only if no guess passed Step 6).**
(a) Re-retrieve with widened context: drop the PO filter, search full warehouse inventory; boost "personal care" using scan history. (b) Re-decode with constraints: beam search may now only output strings from the retrieved candidate list — the model's job becomes "which of these 5 real codes does the image look like most?", which is far easier. (c) Re-verify. One repair round, then fall through to the human.

**Step 8 — Return to user.**
- Auto-accept: phone shows "✓ Dove Soap 100g — received", vibrates, ticks the manifest. ~1 second, zero extra work.
- Confirm: shows top-2/3 with product names + photos ("Dove Soap 100g or Dove Shampoo 180ml?"), worker looks at the box, taps once. System logs the payload and can re-render a clean barcode label.
- Reject: best guess not on this truck at all → "Item not in this shipment — check manually."

**Step 9 — Write-back.**
The confirmed pair (torn-crop embedding → Dove record) is added to the HNSW index. The next torn Dove label anywhere in the warehouse is found instantly by visual retrieval — the system improves with use, no retraining.

---

## USE CASE 2 — Retail cashier / shelf staff (faded, crumpled packaging)

**Step 0 — What the barcode stores.**
Same EAN-13 / UPC number as above — a key into the store's product catalog.

**Step 1 — Offline: build the knowledge base.**
The store's POS catalog: every SKU this store sells (say 30,000 records):
```
payload: 8901491101837 | product: Lays Magic Masala 52g
department: snacks | aisle: 4 | price: ₹20
visual embedding: clean pack + barcode photo
```
Indexed the same three ways. Dynamic context here = which department/aisle the scan happens in (from the store's indoor zones or the POS terminal identity), plus what's currently in the basket (items scanned together are often related).

**Step 2 — Scan + detect.**
Cashier scans a crumpled chips pack; the hardware scanner already beeped "fail." Camera frame → RTMDet crops the wrinkled barcode.

**Step 3 — Neural decode (top-k).**
```
A: 8901491101887  conf 0.55  check ✗
B: 8901491101837  conf 0.52  check ✓
C: 8901491101637  conf 0.31  check ✗
```

**Step 4 — Build the retrieval query.**
Guesses + context (this POS terminal = snacks-heavy front counter; basket already has 2 snack items) + crop embedding.

**Step 5 — Hybrid retrieval + RRF.**
- BM25 over 30k SKUs: `...1837` closest by digits.
- Vector search: crumpled-pack embedding → Lays record nearest (packaging is distinctive even when the barcode is ruined).
- SQL: soft filter/boost on department = snacks.
RRF → Top: `Lays Magic Masala 52g, score 0.89`.

**Step 6 — Verify.**
B: checksum ✓, exists in this store's catalog ✓ → accept.

**Step 7 — Mismatch → repair loop.**
If nothing passes: widen from "snacks" to the full catalog; re-decode constrained to the top retrieved candidates; re-verify. (Retail catalogs are bigger than a manifest, so the visual signal matters more here — that's an interesting per-use-case finding for the paper.)

**Step 8 — Return to user.**
- Auto-accept: item added to the bill: "Lays Magic Masala ₹20 ✓."
- Confirm: shows 2 candidates with pack photos; cashier glances at the pack, taps. Faster than calling a supervisor or price-lookup by name.
- Reject: "Not in catalog — manual entry," e.g. a product this store doesn't stock.

**Step 9 — Write-back.**
Confirmed crumpled-pack embedding stored → future crumpled packs of the same SKU resolve on the visual path alone.

---

## USE CASE 3 — Hospital nurse scanning medication (small, curved, worn vial labels)

**Step 0 — What the barcode stores.**
Usually a 2D DataMatrix with structured text, e.g. GS1 format: `(01)08901234567894(17)270331(10)LOT4432(21)SN998` = drug code + expiry + lot number + serial. DataMatrix includes Reed-Solomon error-correction modules — spare data that recovers the message when part of the grid is destroyed, up to a limit. Beyond that limit, classical decoding fails completely; that's where our decoder + retrieval takes over.

**Step 1 — Offline: build the knowledge base.**
Two layers: (a) the hospital drug formulary (all drugs the pharmacy stocks), (b) the **patient's active medication orders** — for one patient, only 5–10 records:
```
payload prefix: (01)08901234567894 | drug: Amoxicillin 500mg
patient: P-1042 | schedule: 8am/8pm | route: oral
visual embedding: clean vial label photo
```
Dynamic context = which patient's wristband was scanned first. The active retrieval set collapses from thousands of drugs to this patient's handful of orders — the strongest context of all five use cases.

**Step 2 — Scan + detect.**
Nurse scans the patient wristband (sets context = P-1042), then the vial. The label is worn and curved; RTMDet crops and flattens the DataMatrix region.

**Step 3 — Neural decode (top-k).**
RS error correction inside the code has already failed (too many modules destroyed). The decoder outputs:
```
A: (01)08901234567894 (10)LOT4432 ...  conf 0.57  RS partially recoverable ✓
B: (01)08901234567897 (10)LOT4482 ...  conf 0.49  RS ✗
```

**Step 4 — Build the retrieval query.**
Guesses + context (patient P-1042's order list, current time 8:02am → due meds) + crop embedding.

**Step 5 — Hybrid retrieval + RRF.**
- SQL: restrict to P-1042's 7 active orders.
- BM25: guess A's drug code matches the Amoxicillin order.
- Vector: worn-label embedding nearest to the stored Amoxicillin label.
RRF → Top: `Amoxicillin 500mg, due now, score 0.95`.

**Step 6 — Verify (stricter here).**
(a) RS/parity check on guess A ✓. (b) Is the drug in the patient's active orders ✓. (c) Extra safety checks retrieval enables for free: expiry date in the payload not passed ✓; dose time matches schedule ✓.

**Step 7 — Mismatch → repair loop (deliberately limited).**
If the best guess is a drug NOT on this patient's orders: the system NEVER auto-corrects toward the patient's list — that could mask a real wrong-drug event. It re-decodes once against the full formulary only to identify what the vial actually is, then raises a warning. Rule: in safety-critical mode, retrieval is used to verify and warn, never to silently fix. (This asymmetric policy is itself a paper contribution: correction budget as a function of the cost of a wrong answer.)

**Step 8 — Return to user.**
- Auto-accept: "✓ Amoxicillin 500mg — matches order, due 8:00am. Administer."
- Confirm: worn beyond repair → shows the retrieved order candidates with label images; nurse compares with the vial, taps.
- Reject/alarm: "⚠ This appears to be Atorvastatin — NOT on this patient's orders. Do not administer. Verify with pharmacy." The nurse stays inside the safe scanning workflow instead of skipping the scan (the "workaround" problem in BCMA literature).

**Step 9 — Write-back.**
Confirmed worn-label embeddings stored per drug/lot. Curved-vial appearance variations accumulate, improving visual retrieval on exactly the labels that fail most.

---

## USE CASE 4 — Delivery courier (rain-soaked / smudged shipping labels)

**Step 0 — What the barcode stores.**
A Code-128 barcode carrying a tracking ID, e.g. `TRK4482915530IN` (alphanumeric, includes an internal check character), often alongside a QR with the same ID. Again: just a key — the address, recipient, COD amount live in the courier company's system.

**Step 1 — Offline: build the knowledge base.**
Each morning the courier's app already downloads the day's route: 120 packages:
```
payload: TRK4482915530IN | recipient: R. Sharma
address: 12 MG Road | lat/long | COD: ₹0 | stop #37
visual embedding: label photo captured at the hub during sorting (clean)
```
Dynamic context = GPS position + remaining undelivered stops. Standing at one street, only 3–4 packages are plausible. (The hub photo capture is a realistic touch: labels are photographed at induction anyway.)

**Step 2 — Scan + detect.**
At the doorstep in rain, the label is soaked and smudged. RTMDet crops the barcode from the wet, glare-heavy frame.

**Step 3 — Neural decode (top-k).**
```
A: TRK4482915580IN  conf 0.48  check char ✗
B: TRK4482915530IN  conf 0.45  check char ✓
C: TRK4462915530IN  conf 0.40  check char ✗
```

**Step 4 — Build the retrieval query.**
Guesses + GPS (within 200m of stop #37) + remaining-package list + crop embedding.

**Step 5 — Hybrid retrieval + RRF.**
- SQL: undelivered packages within 300m → 3 candidates.
- BM25: `...5530IN` matches 14/15 characters of guess B.
- Vector: wet-crop embedding vs hub-photo embeddings → same package nearest (label layout/handwriting/stamps are distinctive).
RRF → Top: `TRK4482915530IN, R. Sharma, stop #37, score 0.93`.

**Step 6 — Verify.**
B: check character ✓, in today's remaining route ✓, GPS consistent with the delivery address ✓ → accept.

**Step 7 — Mismatch → repair loop.**
If nothing passes: widen radius to the whole route (120), then to the day's hub manifest; re-decode constrained to retrieved tracking IDs; re-verify. Scan history helps too — packages are loaded in stop order, so neighbors of the last delivered stop get boosted.

**Step 8 — Return to user.**
- Auto-accept: "✓ Package for R. Sharma — stop 37. Proof-of-delivery flow opens."
- Confirm: two nearby candidates shown with recipient names; courier checks the printed name on the label, taps.
- Reject: "This package is not on your route — return to hub queue."
No typing a 15-character ID in the rain, no call to the hub.

**Step 9 — Write-back.**
Confirmed wet-label embedding stored against the tracking record for the rest of the day (helps if the same package is rescanned at handover), and degradation examples feed the synthetic-data engine later.

---

## USE CASE 5 — Factory worker scanning DPM parts (codes etched on metal, oily/scratched)

**Step 0 — What the barcode stores.**
DPM = Direct Part Marking: a DataMatrix dot-peened or laser-etched straight onto metal, storing a part serial like `ENG-7742-0093-B`. Includes RS error correction, but oil, scratches, glare, and low contrast on curved metal routinely push damage past what RS can fix. Hardest imaging conditions of all five cases.

**Step 1 — Offline: build the knowledge base.**
The station's work order / BOM for the shift:
```
payload: ENG-7742-0093-B | part: crankshaft assembly
work order: WO-518 | station: 12 | expected sequence #: 93
visual embedding: reference image of this part's marking (from the marking station's QC camera)
```
Dynamic context = station ID + shift work order + scan sequence (parts arrive in near-sequential serials — after `...0092`, serial `...0093` is highly likely).

**Step 2 — Scan + detect.**
Worker scans an oily crankshaft. RTMDet localizes the low-contrast dot pattern on curved metal.

**Step 3 — Neural decode (top-k).**
RS correction inside the code fails (too many destroyed modules). Decoder outputs:
```
A: ENG-7742-0098-B  conf 0.44  RS ✗
B: ENG-7742-0093-B  conf 0.41  RS partially consistent
C: ENG-7742-0083-B  conf 0.39  RS ✗
```

**Step 4 — Build the retrieval query.**
Guesses + context (station 12, WO-518, last scanned = #0092) + crop embedding.

**Step 5 — Hybrid retrieval + RRF.**
- SQL: parts expected at station 12 under WO-518 today (~150 serials).
- BM25: `...0093-B` closest to guesses; sequence prior boosts #0093 strongly (previous scan was #0092).
- Vector: oily-crop embedding vs QC reference images.
RRF → Top: `ENG-7742-0093-B, crankshaft, score 0.91`.

**Step 6 — Verify.**
B: consistent with partial RS parity ✓, exists in WO-518 ✓, fits sequence ✓ → accept.

**Step 7 — Mismatch → repair loop.**
If nothing passes: widen to the plant-level parts database; re-decode with beams constrained to retrieved serials — and here a stronger repair is possible: use the retrieved candidate to *fill in* the uncertain characters, then re-check RS parity of the completed string against the observed modules. RS math turns the retrieved candidate into a hard yes/no, not a guess. If parity now checks out → accept; else next candidate; else human.

**Step 8 — Return to user.**
- Auto-accept: "✓ Part 0093 logged at station 12," traceability record written, line keeps moving.
- Confirm: two serial candidates shown with expected part images; worker compares part type, taps.
- Reject: "Serial not in this work order — possible wrong part routed here. Hold." (catches a real assembly error, not just a scan error).

**Step 9 — Write-back.**
Confirmed oily/scratched marking embeddings accumulate per part type and per marking machine — over weeks the index learns each etching machine's wear signature, catching degradation drift early (a bonus predictive-maintenance angle).

---

## The one pattern behind all five

| Step | Warehouse | Retail | Hospital | Courier | Factory |
|---|---|---|---|---|---|
| Knowledge base | Shipment manifest (200) | Store catalog (30k) | Patient orders (5–10) | Day's route (120) | Work order (150) |
| Dynamic context | PO + dock + scan history | Department + basket | Patient wristband + time | GPS + remaining stops | Station + sequence |
| Verify = math + | "On this truck?" | "In this catalog?" | "On this patient's orders?" | "On my route, near me?" | "In this WO, in sequence?" |
| Repair policy | Auto-correct OK | Auto-correct OK | Warn only, never auto-fix | Auto-correct OK | RS-parity-confirmed fill-in |
| Return to user | Tick manifest | Add to bill | Administer / alarm | POD flow | Traceability log |

Vision appears at four points in every case: detection (find the code), decoding (read top-k), retrieval (crop embedding vs clean references), and repair (re-score image against retrieved candidates). The retrieval set is small and context-selected, which is what makes fuzzy correction safe — and the repair policy changes with the cost of a wrong answer, which is the honest, defensible version of the idea.
