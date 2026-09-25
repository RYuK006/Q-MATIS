# Dedup Cache Test Log

- **Test 1 (Known Duplicates):** Tested 50 structures already in cache.
  - Caught: 50 / 50
  - Time taken: 1.23 seconds (0.0245s per structure)
- **Test 2 (New/Accepted Structures):** Tested 50 non-rejected structures.
  - False positives (marked as duplicate): 12 / 50
  - Time taken: 0.27 seconds (0.0054s per structure)