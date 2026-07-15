# Barcode Recovery and Neural Decoding Project — Complete Slide-by-Slide Model Context

## 0. Purpose of this file

This file converts all 21 photographed slides in Barcode.zip into a single, consistent context document for another AI model. It is intended to be sufficiently complete that the receiving model does not need to inspect the original slide photographs before explaining, reviewing, extending, or presenting the project.

The source slides are marked “Samsung Confidential.” Treat the contents accordingly. The PowerPoint application interface, device address, account name, timestamps, and screen watermark visible around or over the photographed slides are capture artifacts and are not part of the technical proposal.

### Grounding rules for a downstream model

1. Treat statements under “Slide-grounded content” as claims made by the deck, not as independently verified external facts.
2. Preserve reported metrics exactly. Do not silently recompute or correct them.
3. Where slides disagree, retain both claims and use the “Cross-slide inconsistencies and unresolved questions” section to explain the conflict.
4. Do not claim that an item is implemented merely because it appears in the proposed architecture. Distinguish:
   - built or trained;
   - evaluated;
   - proposed;
   - planned for next month;
   - still under training.
5. Use the canonical symbology names in normal prose:
   - Aztec;
   - Code 39;
   - Code 128;
   - EAN-13;
   - Data Matrix;
   - ITF;
   - PDF417;
   - QR;
   - UPC-A.
6. Preserve original spellings when discussing a specific slide. For example, Slide 4 labels Aztec as “Azetic,” which appears to be a typo.
7. Do not collapse the two end-to-end architecture diagrams into one without explaining that they represent different pipeline variants or stages of the research.

---

## 1. Source inventory and slide order

The ZIP contains 21 JPEG photographs and no native PPTX file. The timestamp order is used as the slide order.

| Context slide | Source image | Slide title or primary heading |
|---:|---|---|
| 1 | 20260715_074046.jpg | Problem Definition & Proposed Solution |
| 2 | 20260715_074052.jpg | Summary |
| 3 | 20260715_074055.jpg | DATASET |
| 4 | 20260715_074057.jpg | DATASET — nine barcode symbologies |
| 5 | 20260715_074100.jpg | DATASET — clean, mildly degraded, and strongly degraded examples |
| 6 | 20260715_074102.jpg | DATASET — RTMDet training data and COCO ground truth |
| 7 | 20260715_074105.jpg | Architecture Diagram — Barcode Pipeline Architecture 1 |
| 8 | 20260715_074107.jpg | Architecture Diagram — probabilistic barcode pipeline |
| 9 | 20260715_074110.jpg | Observations — Model Comparison |
| 10 | 20260715_074113.jpg | Results — RTMDet-tiny |
| 11 | 20260715_074115.jpg | Results — Enhancer Results |
| 12 | 20260715_074117.jpg | Results — ViT-NAR versus ViT-AR |
| 13 | 20260715_074123.jpg | Detector |
| 14 | 20260715_074126.jpg | RTMDet Architecture |
| 15 | 20260715_074128.jpg | NAFNet: Enhancer Model |
| 16 | 20260715_074131.jpg | MobileNet v3 + Custom Decoder |
| 17 | 20260715_074134.jpg | Custom Partial Decoder |
| 18 | 20260715_074136.jpg | DONUT (Document Understanding Transformer) |
| 19 | 20260715_074138.jpg | ViT-NAR |
| 20 | 20260715_074141.jpg | ViT-AR — architecture |
| 21 | 20260715_074143.jpg | ViT-AR — overview and symbology classifier |

---

## 2. Executive project narrative

### 2.1 Problem being addressed

The project targets barcode scanning failures in retail and logistics. The deck attributes many failures to blur, physical damage, occlusion, poor or uneven lighting, noise, scratches, tears, ink defects, perspective distortion, and similar degradation.

The deck argues that existing approaches have a three-way trade-off:

- cloud-based commercial recognition can be accurate but introduces latency and privacy concerns;
- conventional on-device libraries such as ZXing and ZBar are fast but can fail completely on degraded barcodes and do not provide useful confidence or partial-decoding output;
- commercial industrial SDKs can be expensive, closed-source, and tied to particular hardware.

The stated gap is an edge-optimized, confidence-scored, multi-format barcode decoder that can recover degraded barcodes while meeting mobile or scanner constraints.

### 2.2 Intended solution

The high-level intended system is a lightweight on-device pipeline:

1. detect one or more barcodes in a full scene using RTMDet-tiny;
2. crop and pad each detected barcode region;
3. attempt to decode the cropped region;
4. if conventional decoding fails, restore or enhance the barcode and decode again, or use a neural decoder;
5. eventually produce top-K payload predictions with confidence scores;
6. use consensus/error correction where multiple probable outputs are available;
7. allow a person to correct low-confidence cases;
8. return the decoded text or payload, barcode format, and bounding box.

The target constraints are:

- under 300 ms latency on an ARM CPU;
- under 200 MB total model footprint;
- support for both 1D and 2D barcodes across nine intended symbologies;
- realistic training data with controlled degradations;
- operation on edge devices rather than dependence on the cloud.

### 2.3 Research progression represented by the deck

The deck records multiple experiments, not one fully settled final model:

- RTMDet-tiny was selected for detection and reportedly achieved more than 99% detection accuracy on the validation set.
- A MobileNetV3 encoder plus U-Net-like decoder was built for image enhancement and produced substantial decode-rate recovery on degraded samples.
- NAFNet was tried as an enhancer, but the deck says it smoothed barcode edges and harmed ZXing decoding.
- DONUT was tried as a neural decoder but was too large and resolution-sensitive, and it failed on 2D formats.
- ViT with a non-autoregressive decoder performed well on some 1D formats but failed on 2D formats because it predicted characters independently.
- ViT with an autoregressive decoder improved 1D results and some 2D results, but QR remained at zero exact decode rate in the shown evaluation. Training is described as ongoing.
- Curriculum training from clean to mildly degraded to strongly degraded data is proposed for the next phase.

### 2.4 Current reported status

Items described as completed or achieved include:

- a synthetic barcode generator;
- a degradation-generation pipeline;
- paired clean/degraded data;
- composite multi-barcode scenes with COCO annotations;
- an RTMDet-tiny detector;
- a MobileNetV3/U-Net enhancement model;
- experiments with NAFNet, DONUT, ViT-NAR, and ViT-AR;
- a trained neural barcode decoder described as ViT + AR decoder;
- detector validation above 99%;
- enhancement from 78.3% baseline decode rate to 94.0% after enhancement.

Items still planned, proposed, or incomplete include:

- stronger curriculum training;
- a stronger image-restoration model for severe degradation;
- final integration of detection, neural decoding, and restoration;
- real-world retail/logistics validation data;
- robust confidence-scored top-K decoding and user correction;
- continued ViT-AR training;
- proof of the under-300-ms ARM CPU latency target.

---

## 3. Canonical component glossary

### RTMDet-tiny

The lightweight detector used to locate barcodes in a full image. It outputs bounding boxes and class/confidence scores. The deck describes it as anchor-free and uses a CSPNeXt-Tiny backbone, CSPNeXtPAFPN neck, and RTMDet separated classification/regression head.

### Crop and pad

The post-detection stage that extracts a detected barcode region and normalizes its spatial framing before decoding or restoration.

### ZXing-CPP

The conventional rule-based decoder used as an initial decoder and as an objective downstream evaluator of restored images. The deck describes ZXing as all-or-nothing: it returns a complete decode or fails, with no partial payload.

### Enhancer or restoration model

An image-to-image model that attempts to turn a degraded barcode crop into an image whose edges/modules are clean enough for a conventional decoder. The main successful enhancer shown is MobileNetV3 plus a U-Net-style decoder. NAFNet was an unsuccessful alternative.

### Neural barcode decoder

A model that maps a barcode image directly to a sequence of payload characters rather than only producing a restored image. The deck evaluates DONUT, ViT-NAR, and ViT-AR.

### ViT-NAR

A ViT-Small image encoder combined with a non-autoregressive transformer decoder. It predicts sequence positions in parallel. The deck attributes its 2D failures to independent parallel predictions that do not capture module/character dependencies.

### ViT-AR

A ViT-Small encoder with an autoregressive transformer decoder. It emits one token at a time, conditioned on previous tokens and image patch features. It also receives a predicted symbology token.

### Symbology classifier

An auxiliary linear classifier that predicts the barcode type from the encoder’s global representation. Its predicted type is passed to the sequence decoder as a conditioning token.

### Decode rate

The percentage of samples whose complete payload exactly matches ground truth. It is all-or-nothing; partial correctness does not count.

### CER

Character Error Rate: edit distance divided by the total number of characters, multiplied by 100. Lower is better. The deck also states character success as 100% minus CER.

### Consensus and error correction

A proposed stage that combines multiple probabilistic decoder outputs to select or reconstruct a more reliable payload. The slides do not specify the algorithm.

---

## 4. End-to-end architecture variants

### 4.1 Pipeline variant 1: conventional decode with restoration fallback

This is the exact logical order shown on Slide 7:

~~~text
Input image
  -> Detect with RTMDet-tiny
  -> Crop and pad the detected barcode
  -> Decode with ZXing-CPP
       -> If successful: output text + format + bounding box
       -> If failed:
            -> Enhance the crop
            -> Decode the enhanced crop with ZXing-CPP
            -> Output text + format + bounding box
~~~

The key design principle is to avoid paying the enhancement cost for easy barcodes. Enhancement is only invoked after the first decoding attempt fails.

### 4.2 Pipeline variant 2: probabilistic neural decoding and consensus

This is the exact logical order shown on Slide 8:

~~~text
Input image
  -> Detection with RTMDet-tiny
  -> Crop and pad
  -> Directed Probabilistic Custom Decoder
       -> Probabilistic Output 1
       -> Probabilistic Output 2
       -> Probabilistic Output 3
  -> Consensus and Error Correction
  -> QR Code Generation
  -> Output (ID / Info / Data)
~~~

Important observations:

- The three probabilistic outputs appear to represent alternatives or top candidates.
- The consensus stage is consistent with the earlier goal of top-K predictions and confidence scores.
- The deck does not define how consensus is calculated, how error correction differs by symbology, or why “QR Code Generation” appears in a multi-format pipeline.
- This should therefore be treated as a proposed research architecture rather than a fully specified implementation.

### 4.3 Detector sub-pipeline

~~~text
3-channel BGR image
  -> resize to 640 x 640
  -> normalize and pad
  -> CSPNeXt-Tiny backbone
  -> CSPNeXtPAFPN multi-scale neck
  -> anchor-free RTMDet head with classification and regression branches
  -> decode box distances
  -> non-maximum suppression
  -> final bounding boxes + class confidence scores
~~~

### 4.4 Image-enhancement sub-pipeline

The successful design is an encoder-decoder restoration network:

~~~text
Degraded 256 x 256 barcode crop
  -> MobileNetV3 hierarchical encoder
  -> multi-scale skip features
  -> U-Net-style trainable decoder with progressive upsampling
  -> convolution + sigmoid output head
  -> restored 256 x 256 barcode image
  -> ZXing decode
~~~

### 4.5 Neural-decoder sub-pipeline

The most advanced shown decoder is:

~~~text
384 x 384 x 3 barcode crop
  -> patch embedding with patch size 8
  -> 2,304 patch tokens
  -> 12-block ViT encoder
  -> symbology classifier predicts one of 9 formats
  -> [BOS, SYM_TOKEN] initializes decoder
  -> 6-layer autoregressive decoder
       - masked self-attention over previous output tokens
       - cross-attention to image patch tokens
  -> linear vocabulary projection
  -> softmax
  -> next token
  -> repeat until the payload sequence ends
~~~

---

## 5. Complete slide-by-slide record

## Slide 1 — Problem Definition & Proposed Solution

**Source:** 20260715_074046.jpg

### Slide-grounded content

#### Problem Domain & Current State-of-the-Art

- Majority of barcode scans fail due to blur, damage, occlusion, or poor lighting.
- Current barcode decoding approaches:
  - Cloud-based examples: Scandit and Anyline. Characterized as accurate but high-latency and privacy-sensitive.
  - On-device examples: ZXing and ZBar. Characterized as fast but unreliable on degraded barcodes and lacking confidence scores.
  - Commercial SDK examples: Cognex and Honeywell. Characterized as expensive, closed-source, and hardware-locked.
- Key gap: no edge-optimized, confidence-scored, multi-format decoder exists.

#### Proposed Solution & Challenges

- A two-stage edge pipeline:
  - Detection using RTMDet-tiny.
  - Neural decoding using a custom decoder.
- The decoder should return top-K predictions with confidence scores.
- A user-in-the-loop should correct low-confidence cases.
- Challenges:
  - less than 300 ms latency;
  - less than 200 MB footprint on an ARM CPU;
  - generalization across nine barcode formats, including both 1D and 2D;
  - realistic training data with controlled degradation.

### Meaning and role in the deck

This slide establishes the product need and the intended novelty: a local, lightweight, confidence-aware system rather than a cloud API, a brittle rule-based reader, or a hardware-tied commercial SDK. It also establishes the difference between mere barcode detection and payload decoding.

### Implementation-status caution

The top-K confidence output and user-in-the-loop correction are stated as proposed features. Later slides show detector confidence and neural-decoder experiments, but do not show a completed top-K UI or a calibrated user-correction loop.

---

## Slide 2 — Summary

**Source:** 20260715_074052.jpg

### Project Goals

- Majority of barcode scans fail in retail/logistics because of blur, damage, or poor lighting.
- Edge devices such as mobile scanners lack the compute needed for heavy vision models.
- Existing solutions are described as:
  - cloud-dependent, causing latency; or
  - rule-based, causing low accuracy.
- Target:
  - real-time recovery;
  - approximately 300 ms on an ARM CPU;
  - model footprint below 200 MB.
- The slide also contains the phrase “An Agent for a agentic platform for Enterprises.” No further agent architecture is explained in this deck.

### List of activities performed or achievements

- Created a degradation pipeline covering blur, noise, occlusion, scratches, cuts, and uneven lighting.
- Generated 35,000 supplementary paired clean/degraded training images.
- Generated 5,000 composite images with COCO annotations for detector training.
- Built RTMDet-tiny to detect barcodes in complex scenes.
- Built a barcode enhancement model using a U-Net-style decoder with a MobileNetV3 encoder.
- Explored multiple decoder architectures: DONUT, ViT, and ResNet.
- Trained a neural barcode decoder using ViT plus an autoregressive decoder.

### Plan for next month

- Curriculum training from clean data toward heavy degradation.
- An image-restoration model.
- Pipeline integration in the order shown on this slide as:

~~~text
Detection -> Decoder -> Restoration
~~~

### Help or support needed

- Real-world barcode scan data from retail/logistics for validation.

### Meaning and role in the deck

This is the status slide. It distinguishes already completed experimentation from near-term work. It also shows that the project still relies primarily on synthetic training data and needs external real-world validation.

### Important ambiguity

The “Detection -> Decoder -> Restoration” sequence conflicts with the fallback order on Slide 7, which performs restoration before the second decode. It may mean “try decoding first, then restore on failure,” but the slide does not explicitly say so.

---

## Slide 3 — Dataset construction

**Source:** 20260715_074055.jpg

### Clean Images

- Reported total: 90,000.
- Composition: 10,000 images per symbology across nine types.
- Libraries:
  - python-barcode for Code 128, EAN-13, UPC-A, ITF, and Code 39;
  - qrcode for QR;
  - pylibdmtx for Data Matrix;
  - aztec_code_generator for Aztec;
  - pdf417gen for PDF417.
- Every clean image was validated with zxing-cpp for 100% decodability.
- Canvas size: 384 × 384 pixels.
- Barcode occupies approximately 85–90% of the canvas.

### Degraded Images

- Reported total: 64,000.
- Includes mild and strong degradation.
- Nine degradation types are applied using Python scripts in degrade.py:

1. **Aging / color fade:** uneven Perlin-noise-based color deterioration.
2. **Ink bleed:** bars become wider and edges softer through dilation plus blur.
3. **Ink dropout:** small holes appear in bars where ink did not stick.
4. **Stains:** coffee, tea, grease, water marks, and dirt, modeled with organic Perlin blobs.
5. **Creases / wrinkles:** highlight and shadow lines cross the barcode.
6. **Realistic tears:** ragged edges and paper-white fill; top/bottom for 1D and corners for 2D.
7. **Scratches:** jagged parallel micro-lines exposing paper and dirt.
8. **Perspective warp:** subtle tilt; the slide says mild mode only.
9. **Paper texture + sensor noise:** Gaussian grain plus a noise overlay.

### Composite Images

- The slide states “16000(clean) + 5,000(degraded)” multi-barcode images with COCO annotations.
- These are used for RTMDet-tiny training.
- Multiple barcodes are placed on complex backgrounds in each image.

### Derived totals, with caution

- If the slide is read literally, the single-barcode pool contains 154,000 images: 90,000 clean plus 64,000 degraded.
- If “16000(clean) + 5,000(degraded)” are separate composite sets, the composite total is 21,000.
- Slide 2, however, says only 5,000 composite images were generated. This conflict must be resolved from the underlying dataset repository or experiment logs before quoting one definitive composite total.

---

## Slide 4 — Supported barcode symbologies

**Source:** 20260715_074057.jpg

### Visual content

The slide displays one clean example of each intended barcode format in a 3 × 3 arrangement:

| Row | Left | Center | Right |
|---|---|---|---|
| Top | “Azetic” image, apparently intended to mean Aztec | Code39 | Code128 |
| Middle | Ean13 | Datamatrix | ITF |
| Bottom | PDF417 | QR | UPC-A |

### Canonical list

1. Aztec.
2. Code 39.
3. Code 128.
4. EAN-13.
5. Data Matrix.
6. ITF.
7. PDF417.
8. QR.
9. UPC-A.

### Meaning and role in the deck

This slide visually confirms the nine-format target and shows that the project is not limited to conventional 1D barcodes. The list spans:

- 1D formats: Code 39, Code 128, EAN-13, ITF, UPC-A;
- 2D formats: Aztec, Data Matrix, PDF417, QR.

PDF417 is technically a stacked 2D symbology; the deck groups the system broadly into 1D and 2D.

---

## Slide 5 — Examples of degradation severity

**Source:** 20260715_074100.jpg

### Visual content

The slide is organized into severity bands:

- **Clean:** a sharp, intact 1D barcode.
- **Mild Degraded:** examples include:
  - a perspective-skewed/tilted 1D barcode with small dropout-like holes;
  - a 1D barcode partially covered by a torn or occluding paper area.
- **Strong Degradation:** examples include:
  - a severely warped or irregular 1D barcode with substantial surface damage;
  - a QR code crossed by many scratch-like lines;
  - a 1D barcode with tears, scratches, and occluded regions.

### Meaning and role in the deck

The slide visually demonstrates why a clean-only training set is insufficient. The strong examples remove or corrupt structure rather than merely adding mild blur. This distinction motivates curriculum training and a decoder capable of using spatial or sequential context.

### Scope caution

The slide provides representative images, not a formal quantitative threshold separating mild from strong degradation.

---

## Slide 6 — RTMDet training data and ground truth

**Source:** 20260715_074102.jpg

### Visual content

The left side shows a synthetic shipping-document background containing:

- two QR codes at different positions;
- one heavily degraded 1D barcode near the bottom;
- varied document words and labels such as shipping/warehouse-style text;
- a complex background intended to approximate a logistics document.

The left caption is **RTMDet Training Data**.

The right side shows a COCO-style annotation object labeled **Ground Truth**:

~~~json
{
  "category_id": 0,
  "bbox": [
    460.0,
    143.00032,
    40.0,
    20.0
  ],
  "area": 800.0,
  "iscrowd": 0,
  "id": 1954,
  "image_id": 349
}
~~~

### Meaning and role in the deck

This slide shows that detector training is not performed only on centered single-barcode crops. Barcodes are composited into multi-object scenes, and COCO annotations provide category IDs and bounding boxes. This is necessary because the detector’s job is to find barcode regions before the decoder handles payload recovery.

---

## Slide 7 — Architecture Diagram: Barcode Pipeline Architecture 1

**Source:** 20260715_074105.jpg

### Slide subtitle

“End-to-End: Image → Detect → Crop → Decode → Enhance → Decode”

### Exact step order

1. **Input Image**
2. **Detect — RTMDet-Tiny**
3. **Crop & Pad**
4. **Decode — ZXing-CPP**
5. Branch on decoder outcome:
   - **Success:** go directly to output.
   - **Fail:** run **Enhance**, then run **Decode — ZXing-CPP** again.
6. **Output — Text + Format + BBox**

### Operational interpretation

- RTMDet locates the barcode.
- Crop/pad isolates and normalizes the region.
- ZXing is used first because it is cheap and reliable on readable samples.
- Enhancement is a conditional fallback.
- ZXing is used again after restoration, providing an objective success criterion.
- The result retains localization information through the bounding box.

### What this diagram does not specify

- which enhancer is used;
- how crops are resized;
- how multiple detected barcodes are scheduled;
- what happens if the second ZXing decode fails;
- whether confidence or a user correction step is returned.

---

## Slide 8 — Architecture Diagram: probabilistic pipeline

**Source:** 20260715_074107.jpg

### Slide subtitle

“End-to-End Pipeline: Image → Detection → Crop & Pad → Probabilistic Decode → Consensus Correction → QR Generation”

### Exact step order

1. **Input Image**
2. **Detection (RTMDet-Tiny)**
3. **Crop & Pad**
4. **Directed Probabilistic Custom Decoder**
5. Three branches:
   - Probabilistic Output 1;
   - Probabilistic Output 2;
   - Probabilistic Output 3.
6. **Consensus & Error Correction**
7. **QR Code Generation**
8. **Output (ID / Info / Data)**

### Operational interpretation

This is the deck’s confidence-aware neural-decoding concept. Instead of a single all-or-nothing ZXing result, the custom decoder produces multiple hypotheses. A later stage reconciles them, applies error correction, and reconstructs a barcode or payload.

### Unresolved design questions

- The decoder architecture that produces the three outputs is not identified on this slide.
- “Directed probabilistic” is not formally defined.
- The consensus rule, confidence calibration, and error-correction procedure are not specified.
- QR-specific generation is shown even though the dataset contains nine formats.
- The meaning of “ID / Info / Data” is not mapped to a formal API schema.

---

## Slide 9 — Observations: Model Comparison

**Source:** 20260715_074110.jpg

### Complete table

| Component | Architecture | Encoder | Decoder | Parameters | Size |
|---|---|---|---|---:|---:|
| DONUT Decoder (2520 × 1980) | Swin-B + BART | Swin-Base | BART | ~225M | ~809 MB |
| ViT-NAR Decoder (384 × 384) | ViT-S + NAR Transformer | ViT-Small (p8) | NAR Transformer | ~28M | ~112 MB |
| ViT-AR Decoder (384 × 384) | ViT-S + AR Transformer | ViT-Small (p8) | AR Transformer | ~28M | ~112 MB |
| Enhancer (MobileNetV3) (256 × 256) | MobileNetV3 + U-Net | MobileNetV3-L | U-Net Decoder | ~3M | ~12 MB |
| Enhancer (NAFNet) (256 × 256) | NAFNet | NAFNet Encoder | NAFNet Decoder | ~17M | ~68 MB |
| Detector (3 × 640 × 640) | RTMDet-tiny | CSPNeXt | RTMDet Head | ~2M | ~8 MB |

### Implications

- DONUT is far above the project’s 200 MB target by itself.
- ViT-AR and ViT-NAR are the same reported size and parameter count; the major difference is sequence-generation behavior.
- MobileNetV3/U-Net is much lighter than NAFNet.
- A detector + MobileNet enhancer + ViT decoder totals approximately 132 MB from the shown rounded sizes: 8 + 12 + 112 MB. This is a derived estimate and excludes runtime, tokenizer, preprocessing, and other binary overhead.
- Replacing MobileNetV3 with NAFNet would produce an approximately 188 MB model-only total: 8 + 68 + 112 MB. This is still below 200 MB only by the slide’s rounded numbers and leaves little deployment overhead.

---

## Slide 10 — RTMDet-tiny detector results

**Source:** 20260715_074113.jpg

### Headline

“RTMDet-tiny: Achieved 99%+ detection accuracy on the validation set.”

### Detector results table

| Metric | Value |
|---|---|
| Model | RTMDet-tiny |
| Input Resolution | 640 × 640 |
| Backbone | CSPNeXt-Tiny |
| Detection Accuracy | 99%+ |
| Training Data | 5,000 composite images |
| Barcode Types | All 8 formats (1D + 2D) |
| Model Size | ~8 MB |

### Key features

- Anchor-free detection head with separate classification and regression branches.
- Multi-scale feature fusion through FPN and PAN paths.
- Non-Maximum Suppression removes duplicate detections.
- Detects multiple barcodes in a single image.
- Lightweight: approximately 2M parameters and approximately 8 MB, presented as suitable for edge deployment.

### Metric caution

The slide calls the metric “detection accuracy” but does not define it as mAP, precision, recall, F1, IoU-threshold accuracy, or another detector metric. Do not relabel it as mAP without supporting experiment logs.

### Format-count caution

This slide says eight formats, while Slides 1, 3, 4, and 21 imply nine. The excluded format is not identified.

---

## Slide 11 — Enhancer results

**Source:** 20260715_074115.jpg

### Headline

“Enhancer Results: 78.3% decode rate on degraded images → 94.0% on enhanced images (+15.6% improvement, 1,409 barcodes recovered).”

The word “improvement” is reported by the slide as 15.6%. Numerically this is a 15.7-point difference using the rounded endpoints, but the deck’s reported value must be preserved.

### Results by barcode type

| Type | Degraded | Enhanced | Reported improvement |
|---|---:|---:|---:|
| code128 | 89.0% | 95.4% | +6.3% |
| ean13 | 93.4% | 96.3% | +3.0% |
| upca | 93.1% | 96.2% | +3.0% |
| qr | 51.0% | 92.5% | +41.4% |
| datamatrix | 51.1% | 95.1% | +44.0% |
| code39 | 74.0% | 85.4% | +11.4% |
| itf | 96.6% | 97.0% | +0.3% |
| **Overall** | **78.3%** | **94.0%** | **+15.6%** |

### Results by degradation type

| Degradation type | Reported improvement |
|---|---:|
| scratch_lines | 24.4% |
| small_occlusion | 18.3% |
| salt_pepper | 17.7% |
| camera_blur_lite | 17.1% |

### Key observations printed on the slide

- 2D barcodes see the largest gains:
  - Data Matrix: +44%;
  - QR: +41.4%.
- 1D barcodes start from a high baseline and show more modest gains, approximately +3–6%.
- Code 39 is the hardest to restore, reaching 85.4%.
- ITF is near-perfect at 97.0%.
- ZXing is all-or-nothing and provides no partial output.
- A neural decoder with confidence scores is needed.

### Labeling issue

The right-side subheading reads “DETECTOR RESULTS,” although the content beneath it is a set of observations about enhancement/decoding. Treat this as a slide-labeling mistake.

### Model attribution caution

The surrounding slide sequence strongly suggests that these are results from the MobileNetV3/U-Net enhancer, but this slide does not explicitly name the enhancer. Use “the reported enhancer” unless experiment logs confirm the exact checkpoint.

---

## Slide 12 — ViT-NAR versus ViT-AR results

**Source:** 20260715_074117.jpg

### ViT-NAR Decoder results

The left table is labeled **ViT-NAR Decoder (1D+2D)**.

| Type | Decode rate | CER |
|---|---:|---:|
| DATAMATRIX | 0 | 79.02 |
| QR | 0 | 57.76 |
| PDF417 | 0 | 61.78 |
| EAN13 | 69.84 | 6.59 |
| UPCA | 72.22 | 6.47 |
| CODE128 | 0 | 50 |
| CODE39 | 98.44 | 0.32 |
| ITF | 91.89 | 0.68 |

The table does not display percent signs consistently, but the surrounding definitions indicate that decode rate and CER are percentages.

### ViT-AR Decoder results

The right table is labeled **ViT-AR Decoder (1D+2D)**.

| Type | Decode rate | CER |
|---|---:|---:|
| Code128 | 90.00% | 3.12% |
| QR | 0.00% | 97.08% |
| DataMatrix | 28.00% | 91% |
| PDF417 | 0.00% | 75.20% |
| Aztec | 0.00% | 52.83% |
| UPC-A | 100.00% | 0.00% |
| ITF | 99.00% | 0.82% |
| Code39 | 100.00% | 0.00% |
| EAN13 | 90.00% | 0.77% |
| **Overall** | **54.11%** | **32.44%** |

### Definitions printed on the slide

- **Decode Rate:** percentage of barcodes decoded completely correctly by exact match. A barcode is either 100% correct or 0%; no partial credit is given.
- **CER (Character Error Rate):** percentage of individual characters that are wrong, calculated as edit distance divided by total characters, multiplied by 100. Lower is better.
- **Character Success:** 100% minus CER.

### Printed conclusion

The slide says that ViT-AR shows significant improvement over ViT-NAR on 1D barcodes:

- Code 128: 0% to 90%;
- Code 39: 98% to 100%.

It also says:

- Data Matrix and Aztec show good results among 2D barcodes;
- QR shows no improvement.

### Careful interpretation

- ViT-AR clearly improves the displayed 1D exact-match rates.
- Data Matrix improves from zero exact matches under NAR to 28% under AR, although the displayed 91% CER remains very high.
- The NAR table does not include Aztec, so the claim that Aztec improves cannot be directly calculated from the two displayed tables.
- QR remains at zero exact-match decode rate and its AR CER is worse than the displayed NAR CER.
- PDF417 remains at zero exact-match decode rate.
- The AR table includes nine formats; the NAR table includes eight.
- No evaluation-set size or confidence interval is shown.

---

## Slide 13 — Detector

**Source:** 20260715_074123.jpg

### Slide-grounded content

- Heading: **RTMDet: Barcode Detection Model**.
- Dataset: a synthetic dataset generated with Python libraries, containing multiple clean and degraded barcodes per image.
- “Negatives were used as text information.” The wording is ambiguous. It may mean text-only regions were included as negative/non-barcode examples, but the slide does not define the negative-sample construction.
- YOLO was considered but dropped because of licensing constraints.

### RTMDet-Tiny architecture diagram

The diagram is labeled “Block-level data flow” and contains these stages:

1. **Input Image:** 3 × 640 × 640.
2. **Preprocessor:** DetDataPreprocessor; normalize and pad.
3. **Backbone:** CSPNeXt-Tiny; four stages plus SPP.
4. **Neck:** CSPNeXtPAFPN; FPN plus PAN.
5. **Head:** RTMDetSepBNHead; classification and regression branches.
6. **Post-Processing:** decode plus NMS.
7. **Output:** bounding boxes plus class scores.

### Meaning and role in the system

The detector performs localization only. It is not responsible for reading the payload. Its output provides the crop coordinates used by the restoration and decoding stages.

### Licensing note

The slide records licensing as a model-selection criterion. It does not identify the precise YOLO version or license, so a downstream model should not generalize the claim to every YOLO implementation.

---

## Slide 14 — RTMDet Architecture

**Source:** 20260715_074126.jpg

### Full component description

| Stage | Component | Function described by the slide |
|---|---|---|
| Input | Image | 3-channel BGR image, resized to 640 × 640 |
| Preprocessor | DetDataPreprocessor | Normalizes pixel values using mean/std and pads to 640 × 640 |
| Backbone | CSPNeXt-Tiny | Feature extractor with four stages plus an SPP bottleneck; pretrained on ImageNet |
| Neck | CSPNeXtPAFPN | Fuses multi-scale features through top-down FPN and bottom-up PAN paths |
| Head | RTMDetSepBNHead | Anchor-free detection head with separate classification and regression branches across three scales |
| Post-Processing | Decode + NMS | Converts predicted left/top/right/bottom distances, written as (l,t,r,b), into bounding boxes; Non-Maximum Suppression removes duplicates |
| Output | Detections | Final bounding boxes with class-confidence scores for the barcode class |

### Step-by-step technical meaning

1. The scene image is converted to a consistent 640 × 640 BGR tensor.
2. Normalization aligns the input distribution with training; padding preserves the required tensor size.
3. CSPNeXt-Tiny extracts increasingly semantic feature maps at several spatial scales.
4. SPP increases the receptive field at the bottleneck.
5. The neck combines high-level semantic features with lower-level spatial detail.
6. The head predicts whether a location corresponds to a barcode and regresses its box offsets.
7. Predicted offsets are converted into explicit boxes.
8. NMS removes overlapping duplicate predictions.
9. Final boxes and confidence scores are passed downstream for crop extraction.

### What is not specified

- confidence threshold;
- NMS IoU threshold;
- data augmentation;
- detection loss;
- optimizer or schedule;
- validation metric definition;
- whether all symbologies share one “barcode” class or use separate classes.

The sample annotation on Slide 6 uses category_id 0, and the output description says “barcode,” which suggests a single detector class, but this remains an inference.

---

## Slide 15 — NAFNet: Enhancer Model

**Source:** 20260715_074128.jpg

### Dataset

A synthetic dataset generated with Python libraries containing moderately degraded barcode images.

### Experiment

Multiple loss functions were explored for the barcode enhancement model.

### Reported problem

The slide characterizes NAFNet as fundamentally an image-smoothing model. It says:

- NAFNet smoothens barcode edges;
- ZXing requires sharp edges for accurate decoding;
- the resulting restored images therefore produced poor decoding performance;
- changing the loss function did not resolve the problem.

### Loss functions

- Official NAFNet loss:
  - L1 loss, or Mean Absolute Error;
  - described as recommended by the paper.
- Other combinations tried:
  - 0.7 SSIM + 0.3 L1;
  - 0.6 × SSIM + 0.1 × L1 + 0.2 × Edge Loss;
  - 0.5 × BCE + 0.3 × Edge Loss + 0.2 × SSIM.

### Definitions printed on the slide

- **SSIM, Structural Similarity Index Measure:** measures image similarity using luminance, contrast, and structure. Range is [0, 1], where 1 means identical.
- **L1 loss:** mean absolute error between the predicted and target images; described as simple and effective for image restoration.

### Important interpretation

Pixel-level perceptual similarity is not the project’s ultimate objective. A restored image can look smoother or more similar to its clean target yet still destroy narrow barcode transitions that ZXing needs. The operative downstream metric should therefore include decodability, not only L1 or SSIM.

### Weighting caution

The second mixed-loss expression shown on the slide sums to 0.9 rather than 1.0. This may be intentional or a transcription/slide omission. Preserve the expression until the training code is checked.

---

## Slide 16 — MobileNet v3 + Custom Decoder

**Source:** 20260715_074131.jpg

### Dataset and loss

- Synthetic dataset generated with Python libraries.
- Contains moderately degraded barcode images.
- Loss:

~~~text
0.5 × BCE + 0.3 × Edge Loss + 0.2 × SSIM
~~~

### Reported performance limitation

- The model performs well on moderately degraded barcodes.
- Under severe degradation such as heavy occlusion, tears, and scratches, it fails completely.
- ZXing is unable to decode the severely degraded output.

### Encoder

The text identifies the encoder as **MobileNetV3-Large**.

- Extracts hierarchical features such as edges and textures from degraded barcode images.
- Provides skip connections at multiple scales to the decoder.

### Decoder

The decoder is a **U-Net decoder**.

- Progressively upsamples feature maps using transposed convolutions.
- Fuses skip connections from encoder stages to recover fine spatial detail.
- Produces a restored barcode image at the original resolution.

### Architecture diagram

The diagram title is **MobileNetV3-Small U-Net Architecture**, which conflicts with the body text’s MobileNetV3-Large label.

The diagram depicts the following approximate tensor progression:

#### Frozen encoder

1. Input: 256 × 256.
2. Encoder Block 1: 16 channels at 64 × 64.
3. Encoder Block 2: 24 channels at 32 × 32.
4. Encoder Block 3: 40 channels at 16 × 16.
5. Encoder Block 4: 576 channels at 8 × 8.

#### Trainable decoder

6. Decoder Block 4: 128 channels at 16 × 16.
7. Decoder Block 3: 64 channels at 32 × 32.
8. Decoder Block 2: 32 channels at 64 × 64.
9. Decoder Block 1: 16 channels at 128 × 128.
10. Final Decoder Block: 16 channels at 256 × 256.
11. Output Head: convolution plus sigmoid.
12. Output: 256 × 256.

The diagram shows encoder-to-decoder skip connections between corresponding scales.

### Interpretation

This architecture is a lightweight restoration model rather than a payload decoder. It succeeds when missing information can be recovered from local/multi-scale visual structure but fails when severe degradation removes too much evidence. That failure motivates the later “custom partial decoder” and neural sequence-decoder work.

---

## Slide 17 — Custom Partial Decoder

**Source:** 20260715_074134.jpg

### Challenge with 2D barcodes

The slide states:

- 1D barcodes mainly require left-to-right encoding information.
- 2D barcodes require complete spatial information.
- 2D data is encoded in a zigzag pattern across multiple modules rather than in a single block.
- A simple encoder therefore cannot handle 2D barcodes effectively.
- A suitable architecture should preserve spatial information and decode the image in patches.

This is the deck’s conceptual motivation. The exact traversal and error-correction structure differs by symbology, so the “zigzag” statement should not be treated as one universal 2D encoding rule without format-specific verification.

### Curriculum training

1. **Stage 1:** clean images.
2. **Stage 2:** mildly degraded images.
3. **Stage 3:** strongly degraded images.

### Meaning and role in the deck

The curriculum is intended to stabilize learning:

- first learn the underlying barcode structure on clean samples;
- then learn robustness to modest corruption;
- finally attempt severe cases.

No epoch allocation, transition criterion, sampling ratio, or stage-specific learning rate is shown.

---

## Slide 18 — DONUT (Document Understanding Transformer)

**Source:** 20260715_074136.jpg

### DonutBarcodeDecoder architecture

1. **Barcode Image:** 960 × 960 RGB.
2. **DonutProcessor:**
   - resize to 480 × 480;
   - normalize.
3. **Swin-B Encoder:**
   - visual feature extraction;
   - output shown as 225 × 1024.
4. **Symbology Classifier:**
   - linear 1024 to 8;
   - predicts barcode type.
5. **BART Decoder:**
   - cross-attends to the encoder;
   - character-level generation.
6. **Output:**
   - decoded payload;
   - symbology;
   - confidence.

The diagram also shows a direct encoder-to-BART connection for visual cross-attention and a symbology-classifier path conditioning the decoder.

### Reported problem

- DONUT is a large model with approximately 225M parameters.
- It was originally trained on high-resolution document images of 2520 × 1980.
- It was tested at lower resolutions of 320, 640, and 960.
- It produced sufficient results only at 960 pixels.
- It worked only for 1D barcodes and failed on 2D formats.

### Implications

- Slide 9 reports approximately 809 MB, so DONUT violates the under-200-MB edge target.
- Its high-resolution requirement also increases latency and memory.
- The resolution notations are internally mixed:
  - Slide 9 labels DONUT as 2520 × 1980;
  - the problem statement says adequate results at 960;
  - the architecture diagram shows a 960 × 960 input resized to 480 × 480.

These may refer to pretraining resolution, evaluation source resolution, and processor tensor resolution respectively, but the slide does not explicitly reconcile them.

---

## Slide 19 — ViT-NAR

**Source:** 20260715_074138.jpg

### Heading

**ViT-NAR (Non-Autoregressive Decoder)**

### Architecture flow

1. Input Image.
2. Patch Embedding.
3. ViT-Small Encoder.
4. Encoder Features.
5. Two branches from the encoder features:
   - direct cross-attention into the NAR Decoder;
   - Symbology Classifier, whose symbology token is fed into the NAR Decoder.
6. NAR Decoder.
7. Decoded Barcode.

### Reported behavior

- Performs well on 1D barcodes.
- The NAR decoder predicts all characters in parallel and independently.
- Because the slide views 2D barcode modules as interrelated, it concludes that parallel independent prediction fails for 2D formats.

### Additional limitation

- Requires a fixed-size input image.

### Interpretation

Non-autoregressive decoding is attractive for latency because all sequence positions can be predicted in one pass. The shown results indicate that the speed-oriented independence assumption sacrifices sequence or spatial consistency, particularly for 2D payloads.

### Result nuance

Slide 12 shows strong NAR performance for Code 39 and ITF, moderate results for EAN-13 and UPC-A, and zero exact decode rate for Code 128 plus all three displayed 2D formats. Therefore “performed well on 1D barcodes” applies to some 1D formats, not all.

---

## Slide 20 — ViT-AR architecture

**Source:** 20260715_074141.jpg

### Heading

**ViT-AR (Autoregressive Decoder)**

### Encoder

1. **Barcode image:** 384 × 384 × 3.
2. **Patch embedding:**
   - patch size 8;
   - approximately 2,304 patches.
3. **ViT encoder:**
   - 12 transformer blocks;
   - six-head self-attention;
   - feed-forward network 384 → 1536 → 384.
4. **Symbology classifier:**
   - linear layer from 384 to 9;
   - predicts the barcode type.

The encoder supplies two types of context to the decoder:

- patch-token memory, shown as [2304, 384], for cross-attention;
- a special sequence initialization containing [BOS, SYM_TOKEN].

### Decoder

1. **Token embedding:**
   - vocabulary size 82;
   - embedding dimension 384.
2. **Six decoder layers:**
   - masked self-attention over prior text tokens;
   - cross-attention where query is text and key/value are image patches;
   - feed-forward network 384 → 1536 → 384.
3. **Linear projection:** 384 → 82.
4. **Softmax:** produces the next-token probability distribution.
5. **Next token:** generated token is fed back to continue the sequence.

### Why it differs from NAR

NAR predicts positions simultaneously. AR predicts the next token using:

- all prior generated tokens;
- image patch features;
- predicted barcode symbology.

This dependency modeling is the central reason the deck expects AR to handle structured payloads better.

### Numeric consistency check

A 384-pixel image divided into 8-pixel patches yields 48 × 48 = 2,304 patches, matching the diagram.

---

## Slide 21 — ViT-AR overview and symbology classifier

**Source:** 20260715_074143.jpg

### AR decoder overview

The slide states:

- tokens are generated sequentially;
- each token is predicted from all previously generated tokens through masked self-attention;
- each token also uses image features through cross-attention with ViT patch tokens;
- the autoregressive approach captures dependencies between characters;
- it showed improvement on 2D barcodes relative to NAR;
- training is currently ongoing.

### Limitation

- Requires a fixed-size input image.

### Symbology classifier

- A linear layer maps 384 features to nine barcode classes.
- It takes the CLS token from the ViT encoder.
- Example classes listed: QR, Data Matrix, and Code 128.

### Purpose of the classifier

- Forces the encoder to learn global barcode structure, including the difference between 1D and 2D.
- During inference, the predicted symbology is passed to the decoder as a token.
- The symbology token conditions character generation so that the output is valid for that barcode type.
- Examples:
  - UPC-A should contain digits only;
  - QR may contain alphanumeric content.

### Status caution

“Showed improvement on 2D barcodes” should be read narrowly:

- Data Matrix improves from zero to 28% exact match in the displayed tables;
- QR remains at zero;
- PDF417 remains at zero;
- an NAR baseline for Aztec is not displayed;
- training is explicitly ongoing.

---

## 6. Consolidated dataset specification

### 6.1 Intended symbologies

| Canonical name | Broad type | Generator named in Slide 3 |
|---|---|---|
| Code 128 | 1D | python-barcode |
| EAN-13 | 1D | python-barcode |
| UPC-A | 1D | python-barcode |
| ITF | 1D | python-barcode |
| Code 39 | 1D | python-barcode |
| QR | 2D | qrcode |
| Data Matrix | 2D | pylibdmtx |
| Aztec | 2D | aztec_code_generator |
| PDF417 | stacked 2D | pdf417gen |

### 6.2 Dataset layers

1. **Clean centered barcode crops**
   - 384 × 384;
   - 90,000 total;
   - 10,000 per intended symbology;
   - validated with zxing-cpp;
   - barcode occupies 85–90% of canvas.

2. **Degraded barcode crops**
   - 64,000 total;
   - mild and strong degradation;
   - derived using nine scripted physical/capture degradations.

3. **Paired clean/degraded training data**
   - Slide 2 reports 35,000 supplementary pairs;
   - likely used for image-to-image enhancement, but exact relation to the 64,000 degraded set is not defined.

4. **Composite detection scenes**
   - multiple barcode crops placed on complex backgrounds;
   - COCO bounding boxes;
   - Slide 2 and Slide 10 report 5,000 images;
   - Slide 3 appears to report 16,000 clean plus 5,000 degraded composites.

5. **Real-world data**
   - not yet shown;
   - requested from retail/logistics for validation.

### 6.3 Degradation taxonomy

| Degradation | Simulated physical/capture effect |
|---|---|
| Aging/color fade | Uneven color deterioration |
| Ink bleed | Wider bars and softened edges |
| Ink dropout | Missing ink holes within bars/modules |
| Stains | Coffee, tea, grease, water, dirt |
| Creases/wrinkles | Highlight and shadow lines |
| Tears | Missing paper regions and ragged boundaries |
| Scratches | Parallel or jagged surface damage |
| Perspective warp | Tilt and projective distortion |
| Paper texture + sensor noise | Grain and imaging noise |

Additional result labels on Slide 11 include scratch_lines, small_occlusion, salt_pepper, and camera_blur_lite. These labels do not map one-to-one onto the nine Slide 3 categories, suggesting that more degradation operators or evaluation tags exist than the summary list shows.

---

## 7. Consolidated model comparison and decision logic

| Model | Role | Reported strength | Reported weakness | Edge suitability |
|---|---|---|---|---|
| RTMDet-tiny | Detect barcode regions | >99% reported validation detection accuracy; multiple objects; ~8 MB | Metric definition and real-world robustness not shown | Strong candidate |
| MobileNetV3 + U-Net | Restore degraded crops | Raises overall ZXing decode rate to a reported 94%; ~12 MB | Fails on severe occlusion, tears, and scratches | Strong candidate for fallback enhancement |
| NAFNet | Restore degraded crops | General restoration architecture; multiple losses explored | Smooths edges needed by ZXing; poor decoding; ~68 MB | Rejected or deprioritized |
| DONUT | Direct neural decoding | Generates payload, symbology, confidence | ~809 MB; high-resolution dependence; fails 2D | Not suitable for target |
| ViT-NAR | Direct neural decoding | Parallel generation; good on selected 1D formats; ~112 MB | Zero exact decode on displayed 2D formats and Code 128; fixed input | Fast but insufficiently robust |
| ViT-AR | Direct neural decoding | Models sequence dependencies; strong displayed 1D gains; some Data Matrix recovery | QR/PDF417/Aztec exact rate still zero in shown table; fixed input; ongoing training | Current neural-decoder direction |
| ZXing-CPP | Conventional decode/evaluation | Fast and reliable on readable inputs | All-or-nothing on degraded inputs; no confidence/partial output | First-pass decoder and evaluation oracle |

### Practical architecture decision implied by the deck

The most plausible edge configuration from the shown experiments is:

1. RTMDet-tiny for localization.
2. ZXing for an inexpensive first decoding attempt.
3. MobileNetV3/U-Net for restoration when ZXing fails.
4. ZXing again after restoration.
5. ViT-AR for cases that still fail, or as the longer-term replacement for all-or-nothing decoding.
6. Confidence/top-K output plus a human correction path for unresolved cases.

Steps 5 and 6 are a synthesis of the deck’s direction, not a fully demonstrated final integrated pipeline.

---

## 8. Consolidated result facts

### Detection

- Reported validation “detection accuracy”: greater than 99%.
- Model: RTMDet-tiny.
- Input: 640 × 640.
- Model size: approximately 8 MB.
- Training data shown in the results table: 5,000 composite images.
- Multiple barcode detection is supported.
- No formal mAP/precision/recall breakdown is shown.

### Enhancement

- Baseline degraded-image ZXing decode rate: 78.3%.
- Post-enhancement decode rate: 94.0%.
- Reported improvement: +15.6%.
- Recovered samples: 1,409 barcodes.
- Strongest format gains:
  - Data Matrix +44.0%;
  - QR +41.4%.
- Strongest listed degradation gain:
  - scratch_lines +24.4%.
- Weakest final format among those shown:
  - Code 39 at 85.4%.
- Highest final format:
  - ITF at 97.0%.

### Neural decoding

- ViT-NAR has excellent displayed results for Code 39 and ITF, moderate results for EAN-13 and UPC-A, and zero exact match for Code 128 and the displayed 2D types.
- ViT-AR reaches:
  - 100% for UPC-A and Code 39;
  - 99% for ITF;
  - 90% for Code 128 and EAN-13;
  - 28% for Data Matrix;
  - 0% for QR, PDF417, and Aztec.
- Overall ViT-AR exact-match rate: 54.11%.
- Overall ViT-AR CER: 32.44%.
- ViT-AR training is ongoing.

---

## 9. Implementation status matrix

| Item | Status supported by slides | Evidence |
|---|---|---|
| Synthetic clean barcode generation | Completed | Slides 2–4 |
| Controlled degradation generator | Completed | Slides 2, 3, 5 |
| COCO composite detector data | Completed | Slides 2, 3, 6 |
| RTMDet-tiny detector | Built and evaluated | Slides 2, 10, 13, 14 |
| MobileNetV3/U-Net enhancer | Built and evaluated on moderate degradation | Slides 2, 11, 16 |
| NAFNet enhancer | Experimented and found unsuitable | Slide 15 |
| DONUT decoder | Experimented and rejected/deprioritized | Slides 9, 18 |
| ViT-NAR decoder | Experimented and evaluated | Slides 9, 12, 19 |
| ViT-AR decoder | Trained/evaluated, still ongoing | Slides 2, 9, 12, 20, 21 |
| Top-K decoder predictions | Proposed | Slides 1 and 8 |
| Consensus and error correction | Proposed diagram only | Slide 8 |
| User-in-the-loop correction | Proposed | Slide 1 |
| Final detector/decoder/restoration integration | Planned | Slide 2 |
| Curriculum clean → mild → strong | Planned or partially defined | Slides 2 and 17 |
| Real-world retail/logistics validation | Needed | Slide 2 |
| Under-300-ms ARM CPU proof | Target only; no benchmark shown | Slides 1 and 2 |
| Under-200-MB model-only configuration | Plausible from rounded component sizes; not deployment-verified | Slides 1 and 9 |

---

## 10. Cross-slide inconsistencies and unresolved questions

These points are important context. A downstream model must not hide them.

### 10.1 Eight versus nine barcode formats

- Slides 1, 3, 4, 20, and 21 indicate nine formats/classes.
- Slide 10 says the detector covers “all 8 formats.”
- The NAR result table lists eight types.
- The AR result table lists nine types.

**Resolution needed:** identify whether one format was excluded from a particular detector/decoder dataset or whether “8” is a stale slide value.

### 10.2 Composite-image count

- Slides 2 and 10: 5,000 composite images.
- Slide 3: appears to state 16,000 clean plus 5,000 degraded composite images.

**Resolution needed:** check dataset manifests and train/validation splits.

### 10.3 Paired-image count versus degraded-image count

- Slide 2: 35,000 paired clean/degraded images.
- Slide 3: 64,000 degraded images.

These can coexist if only a subset is paired or if they are from different iterations, but the relationship is not documented.

### 10.4 Pipeline ordering

- Slide 7: detect → crop → decode → enhance on failure → decode again.
- Slide 2 next-month plan: detection → decoder → restoration.
- Slide 8: detection → probabilistic decoder → consensus → QR generation, with no restoration block.

**Likely interpretation:** the project is comparing a restoration fallback pipeline with a direct neural-decoding pipeline. This should be stated explicitly in future diagrams.

### 10.5 “Two-stage” versus actual number of runtime operations

Slide 1 calls the system two-stage because its two learned tasks are detection and neural decoding. Later diagrams include crop/pad, conventional decode, restoration, consensus, error correction, and output generation. “Two-stage” should not be read as literally only two software operations.

### 10.6 MobileNetV3-Large versus MobileNetV3-Small

- Slide 16 body: MobileNetV3-Large encoder.
- Slide 16 diagram title: MobileNetV3-Small U-Net Architecture.
- Slide 9 table: MobileNetV3-L.

**Most repeated value:** Large, but the actual model configuration must be checked.

### 10.7 DONUT resolution

- Model-comparison table: 2520 × 1980.
- Experiment text: adequate only at 960.
- Architecture diagram: 960 × 960 input resized to 480 × 480.

These may represent original pretraining, test image, and processor tensor sizes, but the slide does not say so.

### 10.8 Enhancement arithmetic

- 78.3% to 94.0% is 15.7 percentage points from the rounded endpoints.
- The slide reports +15.6%.
- Some per-format improvements differ by 0.1 point from simple subtraction.

Use the slide’s reported improvement and note that values may derive from unrounded internal metrics.

### 10.9 NAFNet mixed-loss weights

The expression 0.6 SSIM + 0.1 L1 + 0.2 Edge Loss sums to 0.9. Verify whether a term or coefficient is missing.

### 10.10 Detection metric

“99%+ detection accuracy” is not defined. It must not be presented as mAP without experiment metadata.

### 10.11 “Negatives were used as text information”

The sentence on Slide 13 is unclear. It may mean text regions were used as negative training samples to reduce false barcode detections. Verify the data-generation code.

### 10.12 Enhancer result attribution

Slide 11 does not explicitly name the model. Its position and the broader narrative suggest MobileNetV3/U-Net, but this should be verified.

### 10.13 Symbology classifier output count

- DONUT diagram: linear 1024 to 8.
- ViT-AR: linear 384 to 9.

This probably reflects different experimental format sets.

### 10.14 Claim of 2D improvement

ViT-AR improves Data Matrix but still has zero exact-match rate for QR, PDF417, and Aztec in the displayed table. Future language should say “improved selected 2D results” rather than implying robust 2D decoding overall.

### 10.15 Agentic-platform reference

Slide 2 mentions “An Agent for a agentic platform for Enterprises,” but no agent, tool-use protocol, memory, orchestration, or runtime-agent flow is documented. Treat this as an undeveloped product-context statement.

---

## 11. Missing information needed for a reproducible technical report

The slides do not provide:

- train/validation/test splits;
- random seeds;
- per-format sample counts after degradation;
- payload length distributions;
- background-source licenses or dataset provenance;
- exact definition of mild versus strong degradation;
- degradation probability and parameter ranges;
- data augmentation during model training;
- detector losses, optimizer, learning rate, schedule, epochs, or mAP;
- enhancer optimizer, training schedule, batch size, or checkpoint selection rule;
- exact evaluation-set size for the 1,409 recovered barcodes;
- whether decode rates are macro- or micro-averaged;
- ViT tokenizer and complete 82-token vocabulary;
- sequence maximum length and termination token;
- decoder training objective, label smoothing, beam search, or greedy decoding details;
- top-K value and confidence-calibration method;
- consensus/error-correction algorithm;
- per-device latency, peak RAM, CPU threads, quantization, or runtime framework;
- real-world performance;
- failure handling after enhancer and neural decoder both fail.

These omissions do not invalidate the reported experiments, but they prevent independent reproduction.

---

## 12. Recommended language for future explanations

### Accurate one-paragraph explanation

This project develops an edge-oriented system for detecting and recovering degraded 1D and 2D barcodes. RTMDet-tiny finds barcode regions in complex images, after which a conventional decoder is tried first. Failed crops can be restored by a lightweight MobileNetV3/U-Net enhancer and decoded again, while a ViT-based autoregressive decoder is being developed to provide sequence-aware payload predictions where all-or-nothing decoders fail. Synthetic clean, degraded, and composite COCO datasets support training. The detector reports more than 99% validation detection accuracy, and enhancement raises the reported decode rate from 78.3% to 94.0%, but severe degradation and several 2D formats remain unresolved, real-world validation is still needed, and ViT-AR training is ongoing.

### Accurate novelty framing

The most defensible project-level differentiators are:

1. a lightweight edge pipeline combining detection, conditional restoration, and neural decoding;
2. training data with controllable physical and capture degradations;
3. using downstream barcode decodability, not merely visual similarity, to judge restoration;
4. a symbology-conditioned autoregressive decoder for multi-format payload generation;
5. proposed top-K/confidence output, consensus correction, and human correction for uncertain cases.

Items 4 and 5 should be described with their current maturity: ViT-AR is under training, and the confidence/consensus/user loop is not fully demonstrated in these slides.

### Claims to avoid

Do not say:

- “The system already robustly decodes all nine barcode types.”
- “The detector achieved 99% mAP.”
- “The full pipeline runs under 300 ms on ARM.”
- “Real-world retail performance is validated.”
- “The neural decoder solves QR.”
- “The final consensus/error-correction algorithm is implemented.”
- “NAFNet is universally unsuitable for barcode restoration.”

The slides do not support those claims.

---

## 13. Suggested final integrated runtime, clearly labeled as a synthesis

The following is a coherent integration of the deck’s strongest components. It is not shown as one completed diagram in the source.

~~~text
Full scene
  -> RTMDet-tiny detection
  -> for each detection:
       -> crop + pad + normalize
       -> attempt ZXing-CPP
       -> if ZXing succeeds:
            return payload + symbology + bounding box
       -> if ZXing fails:
            -> MobileNetV3/U-Net restoration
            -> retry ZXing-CPP
       -> if restoration decode still fails:
            -> ViT-AR neural decoder
            -> obtain payload probabilities conditioned on symbology
            -> produce top-K candidates with confidence
            -> run format-aware validity/error-correction checks
       -> if confidence is high:
            return best verified candidate
       -> if confidence is low:
            ask user to choose or correct
       -> if no candidate is valid:
            return explicit failure rather than fabricated payload
~~~

### Why this synthesis fits the deck

- It preserves the cheap success path from Slide 7.
- It uses the successful restoration result from Slide 11.
- It uses ViT-AR’s dependency modeling from Slides 20–21.
- It incorporates the top-K/user-loop goals from Slide 1.
- It incorporates consensus/error correction from Slide 8.
- It gives a defined failure path, which the slides currently omit.

---

## 14. Short chronology of the experiments

1. Define the edge barcode-recovery problem and constraints.
2. Generate clean barcode images across nine symbologies.
3. Validate clean images with ZXing.
4. Create controlled mild and strong degradations.
5. Compose multiple barcodes onto complex backgrounds with COCO annotations.
6. Train RTMDet-tiny for barcode localization.
7. Build a MobileNetV3/U-Net restoration model.
8. Measure whether restoration improves actual ZXing decoding.
9. Try NAFNet; observe smoothing and poor decode behavior.
10. Try DONUT; observe size, resolution, and 2D limitations.
11. Build ViT-NAR; observe strong selected 1D results and poor 2D results.
12. Replace the NAR decoder with an AR decoder and add a symbology classifier.
13. Observe strong 1D gains and partial Data Matrix improvement, while QR/PDF417/Aztec remain unresolved.
14. Plan curriculum learning, stronger restoration, pipeline integration, and real-world validation.

---

## 15. Compact fact block for another model

~~~text
PROJECT:
Edge-oriented degraded-barcode detection, restoration, and neural decoding.

TARGETS:
<300 ms on ARM CPU.
<200 MB footprint.
Nine intended barcode formats, 1D and 2D.
Confidence-aware predictions and human correction for uncertain cases.

DATA:
90,000 clean images (10,000 x 9 formats), 384x384, 85–90% barcode occupancy.
64,000 degraded images.
Nine scripted degradations: aging/fade, ink bleed, ink dropout, stains,
creases, tears, scratches, perspective warp, texture/noise.
35,000 paired clean/degraded images reported on summary slide.
Composite COCO count conflicts: 5,000 versus 16,000 clean + 5,000 degraded.
Real-world retail/logistics data still needed.

DETECTOR:
RTMDet-tiny, 640x640, CSPNeXt-Tiny + CSPNeXtPAFPN + RTMDet head.
~2M parameters, ~8 MB.
Reported >99% validation detection accuracy, metric not formally defined.

ENHANCER:
MobileNetV3 + U-Net, 256x256, ~3M parameters, ~12 MB.
Loss shown: 0.5 BCE + 0.3 Edge Loss + 0.2 SSIM.
Reported decode rate: 78.3% degraded -> 94.0% enhanced.
Reported +15.6%, 1,409 recovered.
Fails under severe occlusion/tears/scratches.

NAFNET:
~17M, ~68 MB.
Rejected/deprioritized because smoothing reduced ZXing decodability.

DONUT:
Swin-B + BART, ~225M, ~809 MB.
Too large/resolution-dependent; worked for 1D, failed 2D.

VIT-NAR:
ViT-Small p8 + NAR transformer, 384x384, ~28M, ~112 MB.
Parallel independent token prediction.
Strong on selected 1D formats; zero exact match on displayed 2D formats.
Fixed-size input.

VIT-AR:
ViT-Small p8 + 6-layer AR decoder, ~28M, ~112 MB.
384x384 image, patch size 8, 2,304 patches.
12 encoder blocks, d=384, 6 heads, FFN 384->1536->384.
Decoder vocab=82, d=384, 6 layers.
Symbology classifier 384->9; supplies SYM_TOKEN.
Overall shown: 54.11% exact decode, 32.44% CER.
Strong 1D; Data Matrix 28%; QR/PDF417/Aztec 0% exact match.
Training ongoing; fixed-size input.

PIPELINE VARIANT A:
Image -> RTMDet -> crop/pad -> ZXing.
On fail -> enhance -> ZXing again.
Output text + format + bounding box.

PIPELINE VARIANT B:
Image -> RTMDet -> crop/pad -> probabilistic custom decoder ->
three candidates -> consensus/error correction -> QR generation ->
ID/info/data output.
Consensus details not specified.

NEXT:
Curriculum clean -> mild -> strong.
Stronger restoration.
Integrate detection, decoder, restoration.
Obtain real-world validation data.
Complete confidence/top-K/user-correction path.
~~~

---

## 16. Final context boundary

This document is exhaustive with respect to the visible content of the 21 supplied slide photographs. It intentionally preserves slide-level wording, reported numbers, architecture order, limitations, and inconsistencies. It does not add external benchmark claims or assume unshown implementation details.
