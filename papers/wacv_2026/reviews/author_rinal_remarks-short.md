**Author Final Remarks - Summary for AC**

**Primary contribution:**

* **Novel, high-resolution, polygon-annotated dataset** for a uniquely difficult small-object category - dog feces - collected with a **before/after/negative (BAN) protocol** to increase negative sample diversity and generalization.
* Distinct from standard benchmarks (amorphous shape, camouflage, extreme sparsity) and **underrepresented in current foundational vision models**.
* Dataset, code, and splits are **openly available** and **hash-verifiable**.

**Main results:**

* Independent test set is small **by design** (no author-collected images) - transparent about limitation.
* Combined with larger validation set, results demonstrate convincing, high-quality detection that:

    * Enables practical applications **right now**.
    * Clearly leaves **headroom for future research** with our dataset.

**Improvements in response to reviews:**

* Added **GroundingDINO** and **YOLOv9** benchmarks, with notable results from GroundingDINO:

  * Zero-shot Box AP **0.23 (test) / 0.08 (val)** -> reveals **domain absence** in current high-quality datasets.
  * Fine-tuned Box AP **0.70 (test) / 0.69 (val)** -> dataset is **sufficiently large** to advance foundational models.
* Reported **F1 and TPR** for both bounding-box and pixel-level results.
* Streamlined tangential sections; retained novel **dataset distribution study**, highlighted by one reviewer for its forward-thinking nature.

**Reviewer context:**

* All reviewers identified similar weaknesses (model diversity, tangential sections, test set size), which we addressed concretely: added models/metrics and streamlined text; test-set size is acknowledged and mitigated by an author-independent design plus results on a larger validation set.
* R1:TVgg - no post-rebuttal response; concerns addressed & main points explicitly resolved.
* R2:Ux3F - initially positive, no concerns after rebuttal.
* R3:NeGf - engaged post-rebuttal; raised score from 4 -> 5.
* R4:LGt7 - engaged post-rebuttal; raised score from 3 -> 4.

**Final position:**

* Reviews began positive and trended upward, with at least two reviewers raising their scores.
* Paper now **meets D&B acceptance standards** - offers a **unique, open dataset**, rigorous baselines, multiple hosting backends, and **new insights into foundational model coverage**.

We thank all reviewers for their feedback, which has resulted in a significantly strengthened and focused paper. 
