# Data: synthetic only

`claims.csv` in this folder is **entirely synthetic**. It was produced by
`python claims_ml.py generate` (see the `generate()` function in `claims_ml.py`), which draws
every field from a random process with a fixed seed (42). There are no real members,
providers, claims, diagnoses, or dollar amounts in it, and it was not derived, sampled,
perturbed, or de-identified from any real dataset.

What was built into the generator so the models have something realistic to learn:

- 10,000 claims over 104 weeks (2024-01-01 to 2025-12-28)
- winter peak and summer trough in volume, +8%/year trend, Christmas / New Year dip
- respiratory diagnoses more common in winter
- plan deductibles that reset in January and burn down through the year
- a slow drift in allowed ratios (contracts get renegotiated)
- about 12% denials driven by out-of-network, late submission, missing prior authorisation
- 3% missing tenure, 2% missing prior-authorisation flag

To use the pipeline on real data, replace this file with an export that has the columns in
README section 9 ("Data dictionary"), keep PHI out of it, and follow README section 10.
