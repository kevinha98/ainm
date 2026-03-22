# Accounting Agent — Learning Log

> Last updated: 2026-03-20

## Current State

- **Status**: 17 regex executors + LLM fallback, deployed rev 00019
- **Revision**: `accounting-agent-00019-bjr`
- **Next action**: Check competition scores → analyze failure patterns → iterate
- **Blocker**: None

## Key Decisions

| # | Decision | Rationale | Date |
|---|----------|-----------|------|
| 1 | Gemini 2.5 Flash as primary LLM | Fast, cheap, good at structured API tasks | 2026-03-19 |
| 2 | Hybrid regex+LLM approach | Regex for speed/efficiency on known tasks, LLM for coverage | 2026-03-19 |
| 3 | Extensive system prompt with API reference | Reduces hallucination and trial-and-error calls | 2026-03-19 |
| 4 | userType="STANDARD" (not "ADMINISTRATOR") | API only accepts "STANDARD" and "NO_ACCESS" | 2026-03-20 |
| 5 | Admin via allowInformationRegistration PUT | "ADMINISTRATOR" is NOT a valid userType | 2026-03-20 |
| 6 | Remove POST fallbacks for action endpoints | Reduces 4xx errors, improves efficiency bonus | 2026-03-20 |

| # | Decision | Rationale | Date |
|---|----------|-----------|------|
| 1 | Gemini 2.5 Flash as primary LLM | Fast, cheap, good at structured API tasks | 2026-03-19 |
| 2 | Single `tripletex_api` tool | Simpler tool = less LLM confusion, max 30 iterations | 2026-03-19 |
| 3 | Extensive system prompt with API reference | Reduces hallucination and trial-and-error calls | 2026-03-19 |

## Prompt Engineering Log

| Date | Change | Impact |
|------|--------|--------|
| 2026-03-19 | Added "action endpoints use PUT not POST" rule | Fixed 405 errors on /:invoice |
| 2026-03-19 | Added "never set vatType on order lines" | Fixed invoice creation failures |
| 2026-03-19 | Added department lookup requirement for employees | Fixed employee creation |
| 2026-03-19 | Added 7-language normalization guidance | German test passing |

## Failed Approaches

| Approach | Result | Date |
|----------|--------|------|
| — | — | — |

## API Quirks Discovered

| # | Quirk | Impact | Date |
|---|-------|--------|------|
| 1 | Action endpoints (/:invoice, /:createCreditNote) use PUT, not POST | 405 if you POST | 2026-03-19 |
| 2 | Order lines: never set vatType (auto-assigned) | Validation error | 2026-03-19 |
| 3 | Employee creation requires department ref + userType: 1 | 422 without them | 2026-03-19 |
| 4 | Fresh accounts have minimal data — create dependencies first | Missing refs | 2026-03-19 |
| 5 | 422 response bodies list exact missing fields | Use for debugging | 2026-03-19 |
| 6 | Some modules need `PUT /v2/company/modules` to enable | Department accounting, project economy | 2026-03-19 |

## Gotchas & Lessons

| # | Lesson | Date |
|---|--------|------|
| 1 | Credit notes for corrections, never delete invoices | 2026-03-19 |
| 2 | Norwegian VAT: 0%(exempt), 8%(food), 12%(transport), 25%(standard) | 2026-03-19 |
| 3 | API response: single=`{value:{}}`, list=`{values:[...]}` | 2026-03-19 |
| 4 | Reference fields use `{"id": N}` pattern | 2026-03-19 |
| 5 | userType only accepts "STANDARD" (needs email) or "NO_ACCESS"; integers 1,2 also work | 2026-03-20 |
| 6 | "ADMINISTRATOR" is NOT valid for userType — use allowInformationRegistration=true PUT | 2026-03-20 |
| 7 | Address must be updated AFTER entity creation — PUT /address/{postalAddress.id} | 2026-03-20 |
| 8 | Always remove read-only fields before PUT: changes, url, displayName, isContact, isProxy | 2026-03-20 |

## Improvement Ideas (Priority Order)

1. **Competition scores** — analyze actual failure patterns from competition submissions
2. **More regex executors** — add handlers for remaining ~13 task types
3. **File processing** — improve PDF/image extraction for tasks with attachments
4. **Pre-fetch common lookups** — departments, VAT types at start → inject into context
5. **Error recovery** — better retry logic for transient 5xx errors
