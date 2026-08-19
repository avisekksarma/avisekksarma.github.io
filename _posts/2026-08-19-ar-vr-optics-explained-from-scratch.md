---
layout: post
title: "AR/VR Optics, Explained From Scratch"
subtitle: "A walkthrough of a hardware survey paper, from waveguides and holographic films to pancake optics."
date: 2026-08-19
categories: [Tech, ar-vr, optics]
tags: ["AR/VR", hardware, optics]
reading_time: 20
description: "AR/VR display optics: field of view, eye-box, vergence-accommodation conflict, waveguides, holographic optical elements, metasurfaces, and pancake lenses."
featured: true
---
> **Paper:** [Choi, Han, Min, et al., "Recent Applications of Optical Elements in Augmented and Virtual Reality Displays: A Review," *ACS Applied Optical Materials* 2(7), 1247-1268 (2024)](https://doi.org/10.1021/acsaom.4c00033).
>
> This is a walkthrough of a **survey paper** on the **hardware and optics of AR and VR displays**. It does not propose one new invention. It rounds up techniques (holographic films, nano-etched surfaces, folded-mirror optics) that engineers use to build headsets, organized around the physical problems that make headsets bulky, narrow-FoV, and headache-inducing. The explanation starts from first principles.

---

## 0. The one idea everything else builds on

Before any of the "advanced" stuff, here's the entire foundation of every AR/VR headset in one sentence:

> **A tiny screen sits very close to your eye, and a lens between the two makes your eye perceive that tiny close-up screen as a huge image far away.**

That's it. Everything else in this post (waveguides, combiners, holograms, metasurfaces, pancake optics) exists purely to make that basic trick **thinner, wider, more comfortable, or more see-through**.

---

## 1. The core problem: why are AR/VR headsets still clunky?

The paper frames the entire field around four physical limitations. Let's go through them one at a time, with an example you can actually picture.

### Problem 1: Field of view vs. eye-box (a tug-of-war)

- **Field of view (FoV)** = how much of your vision the virtual image fills. A narrow FoV feels like looking through a porthole; a wide FoV fills your peripheral vision and feels like you're "in" the scene.
- **Eye-box** = how much your pupil can move around (look left/right, shift slightly) and *still* see the full image without it clipping or going dark.

**Mental experiment: look through a keyhole.** Hold your eye still at a keyhole and you see a decent chunk of the room (that's FoV). Shift your eye even slightly and part of the view goes black (that's a *small eye-box*). If you want to widen the keyhole (more FoV), you'll find you *also* need more room to move your eye around and still catch that wider view: which needs a bigger, heavier optical system.

Physics ties these two together (through something called *étendue*, basically a fixed "budget"): **you can't cheaply increase both FoV and eye-box at once.** That's why cheap headsets either have a narrow FoV, or a wide FoV that clips if you don't hold your head still.

### Problem 2: Vergence-Accommodation Conflict, aka "why VR gives you a headache"

Your eyes use **two separate systems** to judge distance:

- **Accommodation (focus):** the lens inside your eyeball physically reshapes to focus at different distances. Hold a finger near your nose and focus on it: you'll feel your eye straining. Look far away: the strain disappears.
- **Vergence (eye rotation):** your two eyes rotate inward for close objects, and point nearly straight for far ones: this is how your brain triangulates distance.

In real life, these two always agree. **In a VR headset, the physical screen sits at a fixed distance from your eye: always.** So your *focus* system is always locked to that same fixed distance ("optical infinity"), no matter what the content shows. But the *content* can show a 3D scene where something appears to be right in front of your nose, making your eyes converge as if it's close.

Result: your brain gets two contradictory distance signals at once: focus says "far," vergence says "near." This mismatch, sustained over a play session, is the mechanism behind VR-induced eye strain, fatigue, and nausea. Most of the "advanced display techniques" in this paper (Maxwellian view, varifocal, multiplane, light-field, holographic displays) exist specifically to make the *focus distance* match the *content's apparent distance*.

### Problem 3: Bulky and heavy

Remember the foundational trick (tiny screen + lens)? That lens needs a certain physical **gap** between itself and the screen to work: like how a magnifying glass needs to be held a specific distance from the page. That gap is exactly why headsets stick out several centimeters from your face instead of looking like sunglasses.

### Problem 4: "Eye glow"

In some AR designs, some of the display's light leaks out toward other people, so bystanders can see a faint glow around the wearer's eyes: a real social-acceptability problem for wearing these in public/office settings. The paper flags this as still largely unsolved.

---

## 2. Basic vocabulary: building the toolkit

### Waveguide / lightguide: "the periscope trick"

The problem: if the display has to sit directly in front of your eye in a straight line, the whole headset has to stick out. So: **what if the display doesn't have to be in front of your eye at all?**

A waveguide is a **thin, flat piece of glass**. Light enters at one edge (near your temple), and then bounces back and forth *inside* the glass via a real physics effect called **total internal reflection**: the same effect that traps light inside fiber optic cables. Think of being underwater in a pool at night: look up at a shallow angle and the surface acts like a mirror (nothing escapes); look straight up and you see right through. Light inside the waveguide keeps hitting the glass-to-air boundary at an angle shallow enough that it just keeps bouncing, zigzagging down the slab: until it hits a specially engineered spot (the **out-coupler**) positioned right above your eye, which breaks the rule on purpose and lets the light escape downward into your eye.

![Waveguide diagram](/assets/img/ar-vr-optics/01-waveguide.svg)

This is *why* AR glasses like Hololens can look close to normal glasses: the bulky display sits off to the side, and the "delivery pipe" carrying the image to your eye is as thin as a normal lens.

### Combiner: "the see-through mixer"

Specific to **AR** (not VR): you need to see *both* the real world in front of you and a virtual image overlaid on top. A combiner is the component that merges these two light sources into one.

**Everyday example:** look at a lit window at night. You see the room behind the glass *and* a faint reflection of yourself and the street: two different light sources merged by the same piece of glass. A combiner is exactly that, engineered on purpose.

![AR combiner diagram](/assets/img/ar-vr-optics/02-combiner.svg)

Real-world light passes straight through the diagonal glass unaffected; display light bounces off the same glass; both end up traveling in the same direction into your eye, merged into one seamless picture. If the glass were a full mirror, you'd only see the display (real world blocked). If it were fully clear, the display's light would never reach your eye. So a combiner has to hit a careful middle ground: and that balance directly trades off "how vivid is my virtual image" against "how much can I see of the real room" (this is why some AR overlays look faint outdoors in bright daylight).

### Holographic Optical Element (HOE): "a lens that isn't shaped like a lens"

A normal lens bends light because of its **physical curved shape**. An HOE bends light the same way, but it's a **flat, thin recorded film**: no curve needed.

**How you "record" a flat film to act like a curved lens:**

1. Take a light-sensitive film.
2. Shine two laser beams onto it at once: a **reference beam** (plain, boring) and a **signal beam** (shaped exactly like the optical effect you want: e.g. light that's already passed through a real lens).
3. Where the two beams overlap, they create an interference pattern (like ripples from two pebbles dropped in a pond, crossing each other): this pattern gets permanently etched into the film.
4. Later, shine a plain **probe beam** (same as the original reference beam) at the recorded film, and it reconstructs the exact shape of the original signal beam: bending plain light exactly as if it had passed through the real lens.

![HOE recording and playback diagram](/assets/img/ar-vr-optics/03-hoe-record-playback.svg)

**Analogy:** a hologram sticker on a banknote is a flat surface that tricks your eyes into seeing 3D depth. An HOE does the same trick, just engineered to reproduce *lens/mirror behavior* instead of a 3D image.

### Diffraction / Bragg selectivity: "a very picky filter"

An HOE only strongly reacts to light that closely matches the **color** and **angle** of the original recording beam. Light of other colors or from very different angles mostly just passes straight through, ignored.

![Bragg selectivity diagram](/assets/img/ar-vr-optics/04-bragg-selectivity.svg)

**Analogy:** a radio tuned to one station. The antenna is bathed in dozens of frequencies simultaneously, but the circuit only amplifies the one it's tuned to: everything else passes by unnoticed.

Why this matters:
- **Superpower:** because each recording only reacts to its own "key," you can stack multiple different recordings on the *same* physical film without them interfering: like one lock with several independent keyholes.
- **Limitation:** real images use a *range* of colors and a *range* of angles (wide FoV needs light from many directions), but one recording is naturally narrow: tuned to one "station," not a whole band. Much of this paper is engineers fighting this narrowness.

### Metasurface: "same idea, but etched instead of recorded"

Instead of a chemical/laser recording process, a metasurface is manufactured like a computer chip: an array of **nanoscale structures** ("meta-atoms," smaller than the wavelength of light) etched onto a flat surface. Each tiny structure individually delays the light passing through it by a precisely controlled amount.

**Analogy: a marching band forming a curve.** A band standing in a perfectly straight line doesn't need the ground to bend to end up forming a curved arc a few seconds later: you just tell each person to start walking at a slightly different moment. The people in the middle wait; the people at the edges move first. A few seconds later their positions trace a curve, even though they started on a flat line.

![Metasurface diagram](/assets/img/ar-vr-optics/05-metasurface.svg)

That's exactly what's happening: taller/shorter pillars delay light by different amounts, and the *collective* pattern reshapes a flat incoming wave into a converging, focused wave: lens behavior, from a perfectly flat surface.

### Why is HOE even needed? (the actual motivation)

This is worth answering directly, because it ties everything back to Problem 3 (bulk/weight).

**Reason 1: thinness.** A normal lens needs curved, thick glass (real material) to hold its optical power. An HOE gets the *identical* light-bending result recorded into a film thin enough to be a sticker.

![Why HOE thickness comparison](/assets/img/ar-vr-optics/06-why-hoe-thickness.svg)

**Reason 2: a much clearer see-through combiner.** A plain half-mirror combiner permanently reflects a fixed percentage of *all* light, always: dimming the real world by design. Because of Bragg selectivity, an HOE combiner only reacts to the display's specific color/angle; ordinary daylight (broad range of colors and angles) mostly passes straight through untouched. Much better see-through clarity for AR glasses worn all day.

**Reason 3: multiple functions in one layer.** Because HOEs only respond to their own recorded "key," you can stack several optical functions into a *single* thin film: impossible with normal glass, where 3 functions need 3 physically stacked lenses (3x the bulk). This shows up constantly in the paper as "multiplexed" HOEs, e.g. acting like several off-axis mirrors at once to enlarge the eye-box without adding thickness.

**One-line summary:** normal optics force a trade-off between thickness and optical function; HOEs largely break that trade-off: at the cost of only working for a narrow slice of colors/angles.

---

## 3. Section 2 of the paper: Holographic Optical Elements & Metasurfaces

This section is really: *"here are three different recipes for making the recorded/etched film, each with an upgraded superpower, plus what people actually build with them."*

### 3a. Photopolymer HOE (PPHOE): the basic, workhorse version

Recorded into a light-curable plastic (similar idea to 3D-printer resin or UV-cured nail polish). Nothing exotic yet: the "default" recording material. Real jobs people build with it:

1. **Waveguide in/out coupler**: a PPHOE patch glued onto the waveguide slab (from the diagram above) instead of a cut prism.
2. **Free-space combiner**: literally the half-mirror combiner picture, but made of PPHOE film for better see-through clarity.
3. **Lens array**: record a *grid* of tiny lenses side by side (like a bug's compound eye) instead of one big lens. Each tiny lens shows a slightly different viewpoint of the virtual object; your brain fuses these into real depth perception: this is the basis of some light-field 3D displays.

**Downsides:** chromatic aberration (color fringing at edges, because diffraction bends colors differently than normal glass does), and the material physically shrinks a little during curing, throwing off precision.

### 3b. Liquid Crystal HOE (LCHOE): same trick, new superpower: polarization

Same recording idea, but using liquid crystal (the same material inside an LCD screen) instead of plastic. Direct upgrades: wider color/angle range, plus a brand new selectivity: **the "spin direction" of light.**

**Quick explainer on "spin":** light can travel in a corkscrew pattern, spinning either clockwise or counter-clockwise as it moves: this is **circular polarization**, with the two directions usually called **LCP** (left-handed) and **RCP** (right-handed). Think of a left-handed screw vs. a right-handed screw: same shape, opposite twist, and a nut threaded for one doesn't easily thread onto the other.

An LCHOE can be recorded to react *only* to one specific spin, ignoring the other: a third independent "key" on top of color and angle. Two flavors of this behave differently:

**PHOE: simple on/off, based on spin.** Exactly like Bragg selectivity, except the picky trait is spin instead of color/angle. One spin gets the recorded treatment (diffracted); the other is simply invisible to the film and passes straight through.

![PHOE polarization selectivity diagram](/assets/img/ar-vr-optics/07-phoe.svg)

**GPHOE: a smarter version: two different jobs, one per spin.** Instead of "on vs. off," a GPHOE does "job A vs. job B": e.g. LCP light gets converged like a real lens, while RCP light passes straight through as if the film were plain flat glass (a "window").

![GPHOE dual behavior diagram](/assets/img/ar-vr-optics/08-gphoe.svg)

The key difference from PHOE: with PHOE the "wrong" spin is wasted (ignored). With GPHOE, *neither* spin is wasted: both get a meaningful, different job. This unlocks some genuinely clever tricks the paper covers:

- **Switchable focus with no moving parts**: put an electrically-controlled "polarization switch" (a half-wave plate) in front of the GPHOE. Flip it, and the same physical film instantly switches from lens-mode to window-mode.
- **Foveated displays**: switching between lens-mode and window-mode can switch a display between "narrow, sharp, zoomed-in" and "wide, blurry, peripheral" viewing, mimicking how your actual retina works (sharp center, blurry edges): saving a lot of rendering work since you don't need full sharpness everywhere.
- **Eye-box expansion**: toggling spin redirects where the focused image lands, effectively giving your eye more room to move while still catching the image.

### 3c. Metasurfaces: the etched, chip-style version

Same nanostructure idea as before, applied to specific jobs the paper covers:

- **Metalens**: a flat replacement for curved glass lenses; newer "achromatic" versions specifically fix the color-fringing problem that plain HOEs and simple metalenses both suffer from.
- **Metagrating as a waveguide coupler**: the same job as a PPHOE patch on a waveguide, but done with etched nanostructures instead of recorded film, for a wider color/angle range.
- **Dynamic metasurface (a tunable pixel grid)**: wrap each tiny structure with a bit of liquid crystal, and you can electrically tune each one's behavior in real time: turning the surface into a spatial light modulator (SLM), essentially a display where each "pixel" actively reshapes light instead of just switching on/off.
- **3D holographic displays**: metasurfaces designed to project a full 3D image, with tricks (like mixing two polarization states) to reduce "speckle noise": the grainy, shimmery static-like artifact common in laser-based displays, caused by light interfering with itself randomly.

**Downsides:** expensive and hard to manufacture at scale today, and works well only across a limited color range so far: the most "promising but still maturing" end of this field, compared to PPHOE/LCHOE which are already in real prototypes.

### Quick recap table

| Material | Recorded/made how | Extra selectivity | Best for |
|---|---|---|---|
| PPHOE | Laser-recorded in plastic | Color + angle | Cheap, basic waveguide couplers, combiners, lens arrays |
| LCHOE (PHOE/GPHOE) | Laser-recorded in liquid crystal | + spin (polarization) | Switchable focus, foveated views, eye-box tricks, no moving parts |
| Metasurface | Chip-etched nanostructures | Fully custom per-structure | Flat achromatic lenses, tunable displays, cutting-edge 3D holography |

---

## 4. Section 3 of the paper: Combining Optical Elements

### 4a. Plain geometric optics: no holography, just mirrors and lenses

- **Partial mirror / beam splitter**: this is the combiner from Section 2, just built the "boring," non-holographic way.
- **Retroreflector**: bounces light *directly back* the way it came, at almost any incoming angle. You already own several of these: **bicycle reflectors and road signs** work exactly this way, glowing back at car headlights from many angles. In AR displays, this property builds a "Maxwellian view": an image that stays in perfect focus permanently regardless of your eye's own focus state, directly attacking the VAC/headache problem (Problem 2), at the cost of a very small eye-box.
- **Focus-tunable lens/mirror**: a lens or mirror that changes its curve on demand, electronically or via air pressure: basically the same idea as your **phone camera's autofocus**. Used to build "varifocal" displays that shift apparent focus distance in real time to match what you're looking at.

### 4b. Stacking multiple HOEs to cancel out each other's flaws

Since one HOE is good at one thing but bad at another, pair two so each patches the other's weakness. Example from the paper: a PPHOE (decent general lens function, but color-fringy) paired with a GPHOE (color-correcting) builds an 80° FoV AR display with clean color, something neither material achieves alone. It's the optical equivalent of combining two specialized models to cover each other's blind spots.

### 4c. Pancake optics: the technique in your actual headset

This is the most practically important part of the whole paper: it's the real reason modern headsets (Meta Quest 3, Apple Vision Pro) are dramatically thinner than earlier ones (like the original Oculus Rift).

**Core idea:** instead of needing a long straight gap between display and eye (which is what makes a headset thick), **fold that same optical distance back and forth several times inside a tiny gap**, using mirrors.

![Pancake optics folding diagram](/assets/img/ar-vr-optics/09-pancake-optics.svg)

**Analogy:** a periscope-style folded zoom lens on a phone camera: it achieves a long zoom range without a camera bump sticking out, by bouncing the light path inside a tiny module instead of needing one long straight tube.

**How the bouncing works, mechanically:**
- One surface is a **polarizer**: reflects light of one polarization state, lets the other pass (same spin-selectivity idea as GPHOE).
- The other surface is a **curved mirror**: reflects light back *and* gives it optical power (a lens effect), like the retroreflector-plus-curve trick.
- Light bounces between the two 2-3 times; each bounce, a small component quietly flips the light's polarization state a bit. Once enough flips accumulate, the light finally matches the "let it through" condition on the polarizer and exits toward the eye.

So the light travels a much longer effective distance than the physical gap suggests, folded back on itself instead of one long straight corridor.

**The catch:** bouncing isn't free: some light is lost at every bounce. Typical pancake optics only keep ~12.5-25% of the original light (needing brighter displays to compensate), though newer tricks (double-path structures, Faraday rotators) push efficiency up to 50-100%.

---

## 5. The big picture takeaway

Mapping every technique back to the four original problems:

| Original problem | Techniques that attack it |
|---|---|
| **Narrow FoV / small eye-box** (can't have both without bulk) | Multiplexed HOEs, GPHOE eye-box steering, lens arrays, metasurfaces |
| **VAC: eye strain/headaches** (focus distance ≠ apparent distance) | Maxwellian displays (retroreflector trick), varifocal lenses, multiplane / light-field displays |
| **Bulky and heavy** | Waveguides (folding sideways), thin HOE/metasurface films replacing thick glass, pancake optics (folding back-and-forth) |
| **Eye glow / social awkwardness** | Flagged as a real, still largely open problem |

**The paper's actual closing argument, in plain terms:** no single trick wins on every front. Every technique here trades off some combination of field of view, eye-box size, color accuracy, weight, and manufacturing cost against the others. Real progress comes from **smart combinations**: pairing a PPHOE with a GPHOE to cancel color problems, pairing a waveguide with a combiner for thin see-through AR, pairing polarization switching with curved mirrors for pancake optics. The authors' bet is that getting to "AR glasses that feel like normal eyeglasses" needs continued collaboration between materials scientists (better recording chemicals / nanostructures), optical engineers (clever geometric combinations), and software/rendering people (foveated rendering, gaze tracking, content pipelines that exploit all this hardware).

---

## Glossary (quick lookup)

- **NED**: Near-eye display, the general term for any headset display.
- **FoV**: Field of view; how much of your vision the virtual image fills.
- **Eye-box**: The area your pupil can move within and still see the full image.
- **VAC**: Vergence-Accommodation Conflict; the mismatch between focus distance and perceived distance that causes VR eye strain.
- **Waveguide / lightguide**: Thin glass slab that carries light sideways via total internal reflection.
- **Combiner**: Optical element that merges real-world light with display light (AR only).
- **HOE**: Holographic Optical Element; a flat film recorded to bend light like a lens/mirror.
- **Bragg selectivity**: An HOE's pickiness: it only reacts to light matching its recorded color/angle.
- **PPHOE**: HOE recorded in photopolymer (plastic).
- **LCHOE**: HOE recorded in liquid crystal; adds polarization (spin) selectivity.
- **PHOE**: Polarization-dependent HOE; on/off behavior based on spin.
- **GPHOE**: Geometric-phase HOE; does two different jobs, one per spin (e.g. lens vs. window).
- **Metasurface**: Etched array of nanostructures ("meta-atoms") that collectively bend light like a lens, without any curve.
- **Metalens**: A metasurface built specifically to act as a lens.
- **SLM**: Spatial light modulator; a device (or tunable metasurface) that can reshape light pixel by pixel in real time.
- **Maxwellian view/display**: A display technique using a pinhole/retroreflector trick to keep the image always in focus, regardless of the eye's own focus state.
- **Varifocal display**: A display where the focus distance changes dynamically to match content depth.
- **Foveated display**: A display that renders the center of vision sharply and the periphery coarsely, mimicking the human retina.
- **Pancake optics**: Folding the optical path back and forth between two close mirror surfaces, to shrink headset thickness.

---

*Diagrams are original illustrations for this walkthrough, not reproductions from the paper.*
