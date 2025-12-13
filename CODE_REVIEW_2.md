# Code Review 2 - Dead Code, Fallbacks, and Duplication

Review date: 2025-12-13

Focus areas:
- Dead code or deprecated code (whole methods or fallbacks never hit)
- Fallbacks that could mask errors
- Code duplication or reuse opportunities

---

## 1. Dead Code

### High Priority - Safe to Remove

| Location | Function/Class | Evidence | Lines |
|----------|---------------|----------|-------|
| `Moisturizer.py` | `get_live_prediction_data()` | Only 1 occurrence (definition), never called | ~88 |
| `Moisturizer.py` | `get_sequence_data()` | Only 1 occurrence (definition), never called | ~90 |
| `create_moisture_map.py:830` | `get_elevation_from_open_elevation()` | Marked `DEPRECATED`, only 1 occurrence (definition) | ~28 |
| `augmented_live.py:823` | `benchmark_dataset()` | Only 1 occurrence (definition), never called | ~36 |
| `precompute_augmented.py:199` | `_process_samples_worker_v2()` | Alternative worker function, never called. Superseded by `_process_batch_direct_write()` | ~97 |

**Estimated removal: ~339 lines**

### Medium Priority - Consider Removing

| Location | Function/Class | Notes |
|----------|---------------|-------|
| `augmented_live.py:678` | `AugmentedPrecomputedDataset` class | Only used in `augmented_live.py` itself (5 internal refs) and test files. If precomputed augmented datasets are loaded via `loadDataset()`, this class may be redundant. | ~143 lines |
| `create_moisture_map.py` | `debug_find_worst_offenders()` | Debug function (184 lines). Consider removing if no longer used for development. |

---

## 2. Error-Masking Fallbacks

### High Risk - Bare `except:`

| Location | Code | Issue | Recommendation |
|----------|------|-------|----------------|
| `Moisturizer.py:474` | `except: return False` | Catches ALL exceptions including `KeyboardInterrupt`, `SystemExit`. Could mask programming errors. | Change to `except Exception:` or more specific exceptions, and consider logging. |

```python
# Current (line 474):
except:
    return False

# Recommended:
except (requests.RequestException, KeyError, ValueError) as e:
    # Optionally log: print(f"Warning: soil moisture check failed for station: {e}")
    return False
```

### Medium Risk - Silent Fallbacks

These `except Exception` blocks print warnings but continue, which could mask systematic issues:

| Location | Context | Behavior |
|----------|---------|----------|
| `create_moisture_map.py:126` | Loading admin boundaries | Falls back to land data silently |
| `create_moisture_map.py:793` | Loading SRTM elevation | Returns `None, None, None` - callers must handle |
| `Moisturizer.py:856` | Loading filtered_params cache | Continues without cache (acceptable) |
| `Moisturizer.py:923` | Saving filtered_params cache | Prints warning (acceptable) |
| `Moisturizer.py:2766` | Validating dataset cache | Clears timeseries_df (acceptable) |

The fallbacks at lines 856, 923, and 2766 are reasonable for cache operations. The create_moisture_map.py fallbacks could benefit from more structured error handling if reliability is critical.

---

## 3. Code Duplication

### Skip Pattern Generation (Duplicated 3x)

The same skip pattern generation logic appears in three places:

**Location 1: `augmented_live.py:330-344`**
```python
def _build_augmentation_patterns(self):
    available_indices = list(range(self.n_nearby_available))
    self.skip_patterns = []
    if self.n_nearby_available > self.n_nearby_in_features:
        for skip_idx in range(self.n_nearby_available):
            keep_indices = [i for i in available_indices if i != skip_idx][:self.n_nearby_in_features]
            self.skip_patterns.append(keep_indices)
    else:
        self.skip_patterns.append(list(range(self.n_nearby_in_features)))
```

**Location 2: `precompute_augmented.py:335-344` (in `_setup_augmentation`)**
**Location 3: `precompute_augmented.py:708-717` (in `generate_all_augmentations_batched`)**

Both locations in `precompute_augmented.py` have identical logic.

**Recommendation**: Extract to a shared utility function:

```python
# In Moisturizer.py or a shared utils module:
def build_skip_patterns(n_nearby_available: int, n_nearby_in_features: int) -> List[List[int]]:
    """
    Build skip patterns for augmentation.

    Returns list of index lists, where each list contains indices of stations to keep.
    """
    available_indices = list(range(n_nearby_available))
    skip_patterns = []
    if n_nearby_available > n_nearby_in_features:
        for skip_idx in range(n_nearby_available):
            keep_indices = [i for i in available_indices if i != skip_idx][:n_nearby_in_features]
            skip_patterns.append(keep_indices)
    else:
        skip_patterns.append(list(range(n_nearby_in_features)))
    return skip_patterns
```

---

## 4. Summary

### Recommended Actions

1. **Remove dead code** (~339 lines):
   - `get_live_prediction_data()`
   - `get_sequence_data()`
   - `get_elevation_from_open_elevation()` (deprecated)
   - `benchmark_dataset()`
   - `_process_samples_worker_v2()`

2. **Fix bare except** at Moisturizer.py:474

3. **Extract shared function** for skip pattern generation to eliminate 3x duplication

4. **Consider removing**:
   - `AugmentedPrecomputedDataset` class if not used in production
   - `debug_find_worst_offenders()` if no longer needed for development

### Impact

- **Lines removed**: ~339-500+ depending on decisions
- **Risk**: Low (all identified code is unused or deprecated)
- **Maintenance benefit**: Reduced cognitive load, clearer codebase
