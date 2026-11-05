# Physical Accessibility Grading Rubric for the 600 Audited Urban Attractions

To ensure empirical realism across diverse urban morphologies—including historic mountain temples, medieval cobblestone quarters, and modern barrier-free civic institutions—each of the 600 cataloged POIs was graded on a 3-tier physical accessibility scale (`accessibility_tier` $\in \{0, 1, 2\}$) cross-referenced against OpenStreetMap `wheelchair=*` tags, municipal accessibility guides, and official venue visitor-access statements (snapshot date: March 15, 2026):

1. **Tier 2 — Full Step-Free / ADA-Equivalent Access (`accessibility_tier = 2`, binary `wheelchair_accessible = 1`)**:
   - Continuous step-free entrance and primary exhibition/viewing route via permanent ramps, elevators, or level paving.
   - Accessible restrooms on site and barrier-free drop-off or public transit connection.
   - Prevalence across corpus: **47.3%** overall (**59.2%** in Info-Rich megacities with extensive modern civic infrastructure vs. **29.6%** in Info-Scarce historic/mountainous secondary cities such as Quanzhou, Dali, Porto, and Chiang Mai).

2. **Tier 1 — Partial or Assisted Access (`accessibility_tier = 1`, binary `wheelchair_accessible = 0`)**:
   - Ground-level courtyard or exterior plaza is step-free, but interior historic halls, upper floors, or secondary trails require steps or staff-operated portable lifts.
   - Cobblestone or uneven historic stone paving navigable only with assistance or all-terrain mobility aids.

3. **Tier 0 — Unmodified Stepped or Steep Heritage Terrain (`accessibility_tier = 0`, binary `wheelchair_accessible = 0`)**:
   - Steep mountain trails, cliffside grottoes, medieval watchtowers, un-elevatored pagodas, or stepped historic alleys where structural preservation precludes ramp installation.
