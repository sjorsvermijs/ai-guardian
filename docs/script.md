# AI Guardian — 3-Minute Demo Video Script

**Target**: MedGemma Impact Challenge submission (3-minute max)
**Runtime**: ~2:55 (5s buffer)

---

## Act 1: Joy + Tension (0:00 - 0:25)

**Visual**: Happy footage — a baby being born, first smile, parents laughing. Warm, golden tones. Upbeat soft music.

> *Text on screen*: "The happiest moment of your life."

**Visual**: Smash cut — 6 months later. Night time. Baby crying. Parent wakes up, worried face lit by phone screen.

**Voiceover**:

> Six months later. It's 2 AM. Your baby has a fever, a rash, and won't stop crying. Is this an emergency — or can it wait until morning?

> *Text on screen*: "What if your phone could help you decide?"

---

## Act 2: Demo 1 — Show It Working (0:25 - 1:15)

**Visual**: Screen recording. Parent opens AI Guardian, uploads a video of their fussy baby. No explanation yet — just let the viewer watch.

**Voiceover**:

> Meet AI Guardian. Record 10 seconds of your baby. That's it.

**Visual**: Processing animation runs. Results appear one by one — heart rate, cry classification, respiratory assessment, triage level.

**Voiceover**:

> From a single video, AI Guardian extracts your baby's heart rate and breathing from invisible skin color changes... identifies the cry pattern... screens respiratory sounds... and generates a personalized report.

**Visual**: Zoom into the parent message — warm, reassuring language.

**Voiceover**:

> A message for you, in plain language.

**Visual**: Zoom into the specialist message — clinical terminology.

**Voiceover**:

> And a clinical note your pediatrician can use tomorrow morning.

**Visual**: Zoom into triage badge showing "LOW".

**Voiceover**:

> Priority: low. You can breathe. It can wait until morning.

---

## Act 3: How It Works (1:15 - 1:55)

**Visual**: Clean animated diagram — one video splits into four parallel pipelines, converging into MedGemma.

**Voiceover**:

> Here's what just happened. One video was analyzed by four AI pipelines running in parallel.

**Text on screen** (appear one by one with icons):
- rPPG — Heart rate, breathing, oxygen from video
- Cry Analysis — What type of cry? Hunger, discomfort, pain?
- Google HeAR — Respiratory sound screening
- MedGemma 1.5 — Skin condition detection

**Voiceover**:

> All four results are fused by MedGemma 4B — Google's medical AI — which reasons like a pediatrician. It cross-validates findings against WHO and NICE clinical guidelines built into the system, and generates the triage report.

> *Text on screen*: "3 Google Health AI models. WHO & NICE guidelines. Zero cloud. Runs entirely on your device."

---

## Act 4: Demo 2 — Skin Condition (1:55 - 2:20)

**Visual**: New screen recording — different video, baby with visible spots on skin.

**Voiceover**:

> But what happens when something is actually wrong?

**Visual**: Results appear — VGA shows "chickenpox", triage escalates to MODERATE.

**Voiceover**:

> MedGemma 1.5 spots the chickenpox from video screenshots. The triage level jumps to moderate. The parent message changes tone — see a doctor today.

**Visual**: Zoom into updated parent message, now more urgent.

**Voiceover**:

> And the system is honest about what it doesn't know. When audio quality is poor, it says inconclusive — not a false alarm.

---

## Act 5: Why It Matters + Close (2:20 - 2:55)

**Visual**: Montage — parent looking relieved at phone. A rural village. A phone in someone's hand in a place with no hospital nearby.

**Voiceover**:

> AI Guardian isn't replacing doctors. It's the triage nurse that every parent deserves at 2 AM.

> For the 2 billion people without easy access to a pediatrician, a phone is often the only medical device they have. We're making it smarter.

**Visual**: Fade to closing card.

**Text on screen**:

> **AI Guardian**
> Because every parent deserves a second opinion.
>
> Built with MedGemma 4B, MedGemma 1.5, and Google HeAR
> 100% local. 100% private. Zero cloud.
>
> MedGemma Impact Challenge 2026

---

## Full Voiceover Script (continuous)

*Read this straight through for recording. ~2:50 at natural pace.*

Six months later. It's 2 AM. Your baby has a fever, a rash, and won't stop crying. Is this an emergency — or can it wait until morning?

Meet AI Guardian. Record 10 seconds of your baby. That's it.

From a single video, AI Guardian extracts your baby's heart rate and breathing from invisible skin color changes... identifies the cry pattern... screens respiratory sounds... and generates a personalized report.

A message for you, in plain language. And a clinical note your pediatrician can use tomorrow morning.

Priority: low. You can breathe. It can wait until morning.

Here's what just happened. One video was analyzed by four AI pipelines running in parallel.

All four results are fused by MedGemma 4B — Google's medical AI — which reasons like a pediatrician. It cross-validates findings against WHO and NICE clinical guidelines built into the system, and generates the triage report.

Three Google Health AI models. WHO and NICE guidelines. Zero cloud. Runs entirely on your device.

But what happens when something is actually wrong?

MedGemma 1.5 spots the chickenpox from video screenshots. The triage level jumps to moderate. The parent message changes tone — see a doctor today.

And the system is honest about what it doesn't know. When audio quality is poor, it says inconclusive — not a false alarm.

AI Guardian isn't replacing doctors. It's the triage nurse that every parent deserves at 2 AM.

For the 2 billion people without easy access to a pediatrician, a phone is often the only medical device they have. We're making it smarter.

---

## Production Notes

- **Music**: Soft piano/ambient for Acts 1 and 5, minimal or none during demos
- **Screen recordings**: Record at 1080p, zoom into key UI elements. Consider a phone mockup frame around the browser to reinforce "just your phone"
- **Demo videos**: Use real baby videos (with consent) or stock footage for emotional scenes. App demos must be real screen recordings.
- **Demo 2 (skin)**: VGA is disabled on 8GB Macs due to OOM. Options: (a) record on a 16GB Mac with VGA enabled, (b) run standalone `inference_mlx.py` and composite the result into the UI
- **Text overlays**: Keep on screen for at least 3 seconds. Use consistent font and placement.
- **Pacing**: The voiceover has natural pauses built in. Don't rush — silence during the demo lets the viewer absorb the UI.
