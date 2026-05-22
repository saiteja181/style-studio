# Test selfie credits

> **TEST FIXTURES ONLY — DO NOT DEPLOY TO CUSTOMERS.**
>
> These photos are licensed for engineering acceptance testing only.  Pexels'
> license permits commercial use of the photo but explicitly does NOT grant
> model-release rights to depicted subjects.  Showing these specific photos
> to a paying salon customer (preview screen, marketing material, app
> screenshots) or including them in any production deployment artefact
> creates real legal exposure under Indian publicity-rights statutes and
> equivalents in other jurisdictions.
>
> Before any customer-facing ship, replace these fixtures with photographer-
> contracted release-cleared portraits (₹500-2,000 per shot in India).  Or
> route the salon's own first-customer real photo through the pipeline once
> a release form is signed.

The two acceptance fixtures used by `tests/run_acceptance_test.py` were
sourced under permissive licenses.  This file records attribution so the
project complies with each source's terms.

## young_indian_man.jpg
- Source: https://www.pexels.com/photo/positive-bearded-young-indian-guy-in-turban-4307869/
- Photographer: Ketut Subiyanto (https://www.pexels.com/@ketut-subiyanto/)
- License: Pexels free-to-use (commercial and non-commercial; attribution appreciated, not required)
- Downloaded: 2026-05-23

## young_indian_woman.jpg
- Source: https://www.pexels.com/photo/19011883/
- Photographer: Chalta Phirta (https://pexels.com/@chalta-phirta-307182428/)
- License: Pexels free-to-use (commercial and non-commercial; attribution appreciated, not required)
- Downloaded: 2026-05-23

## round_face_indian_man.jpg
- Source: https://www.pexels.com/photo/portrait-photography-of-man-smiling-2324638/
- Photographer: Yogendra Singh (https://pexels.com/@yogendras31/)
- License: Pexels free-to-use (commercial and non-commercial; attribution appreciated, not required)
- Downloaded: 2026-05-23

## curly_hair_indian_woman.jpg
- Source: https://www.pexels.com/photo/portrait-of-a-fashionable-woman-with-long-hair-31541803/
- Photographer: Anil Sharma (https://pexels.com/@shootsaga/)
- License: Pexels free-to-use (commercial and non-commercial; attribution appreciated, not required)
- Downloaded: 2026-05-23

## dark_skin_indian_man.jpg
- Source: https://www.pexels.com/photo/a-man-in-blue-and-gray-plaid-shirt-7529112/
- Photographer: T K Dhamu (https://www.pexels.com/@tkdhamu/)
- License: Pexels free-to-use (commercial and non-commercial; attribution appreciated, not required)
- Downloaded: 2026-05-23
- Note: not in current acceptance CASES; reserved for future demographic-coverage run.

## Legacy fixtures
- `test_random_indian_man.jpg` (watermarked, no longer acceptance gate)
- `test_indian_woman_a.jpg` (tight portrait, blocked by pre-flight)
- `test_indian_woman_b.jpg` (older grey-haired woman, no longer acceptance gate)

These remain on disk for historical reference but are not used by current
acceptance tests.  Delete in a future cleanup if storage matters.
