# AI-4E — Investigation Intelligence

AI-4E is a conservative, clinician-facing longitudinal investigation review layer.

## Endpoint

`GET /api/patients/{patient_id}/investigation-intelligence`

Requires an authenticated `doctor`, `triage`, or `admin` user.

## Behavior

- Uses only explicitly verified document extraction items.
- Groups repeated investigation labels and reports descriptive numerical changes.
- Flags differing recorded units instead of silently converting them.
- Keeps evidence, document IDs, filenames, and date basis for traceability.
- Does not call a value normal/abnormal without a reliable reference range.
- Does not diagnose, prescribe, or recommend treatment.
- If an explicit test date is unavailable, document creation time is clearly labelled as the date basis.

## Clinical safety boundary

A trend such as `decreased` is a mathematical observation, not a clinical interpretation. Significance must be determined by a clinician using the original report, reference range, units, patient context, and local protocol.
