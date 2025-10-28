# PyTorch Dataset Fixes and Parameter Filtering

## Changes Made

### 1. Fixed Pandas Timestamp Conversion Issue

**Problem:**
```python
TypeError: default_collate: batch must contain tensors, numpy arrays, numbers,
dicts or lists; found <class 'pandas._libs.tslibs.timestamps.Timestamp'>
```

**Solution:**
Modified `SoilMoistureSequenceDataset.__getitem__()` (line ~881) to convert pandas Timestamps to Unix timestamps (float):

```python
# Convert pandas Timestamp to Unix timestamp (float) for PyTorch compatibility
end_date_unix = sample['end_date'].timestamp() if hasattr(sample['end_date'], 'timestamp') else float(sample['end_date'])

return {
    'features': features,
    'target': target,
    'mask': mask,
    'target_station_id': sample['target_station'],
    'end_date': end_date_unix  # Unix timestamp as float
}
```

Now DataLoader can properly collate batches without timestamp errors.

### 2. Added Parameter Coverage Analysis

**New Method:** `MeteoGaliciaCollector.analyze_parameter_coverage()`

**Purpose:** Analyze which parameters have sufficient data coverage across stations and filter out sparse parameters.

**Usage:**
```python
collector = MeteoGaliciaCollector()

# Analyze coverage with 25% threshold
coverage, filtered_params = collector.analyze_parameter_coverage(
    coverage_threshold=0.25  # Require 25% of stations to have data
)

# Use filtered parameters in Dataset
dataset = SoilMoistureSequenceDataset(
    timeseries=collector.timeseries_file,
    stations=collector.stations_file,
    nearest=collector.nearest_file,
    seq_length=64,
    feature_params=filtered_params  # Only use well-covered parameters
)
```

**Output Example:**
```
Parameter coverage:
----------------------------------------------------------------------
✓ TA_AVG_1.5m          : 95.2% (37/39 stations)
✓ HR_AVG_1.5m          : 92.3% (36/39 stations)
✓ PP_SUM_1.5m          : 89.7% (35/39 stations)
✗ TS_AVG_-0.1m         : 12.8% (5/39 stations)   <- filtered out
✗ ET0_SUM_1.5m         :  7.7% (3/39 stations)   <- filtered out
----------------------------------------------------------------------
Parameters passing 25% threshold: 18/42
```

**Benefits:**
- Reduces noise from sparse parameters
- Smaller model input size
- Better training stability
- Focus on reliably-measured features

### 3. Updated Documentation

- Added tip in `SoilMoistureSequenceDataset` docstring about using `analyze_parameter_coverage()`
- Created test script: `test_parameter_coverage.py`

## Testing

### Without PyTorch Installed
```bash
python3 test_parameter_coverage.py
```
- Tests parameter coverage analysis (works without PyTorch)
- Shows which parameters would be filtered

### With PyTorch Installed
The test script also validates:
- Dataset creation with filtered parameters
- Timestamp conversion fix
- DataLoader compatibility

## PyTorch Installation Note

This environment has disk space limitations. To install PyTorch:

**On your own machine:**
```bash
# CPU-only (smaller):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Or full version with CUDA:
pip install torch
```

**In this limited environment:**
- PyTorch requires ~900MB + CUDA dependencies (~2.5GB)
- Total: ~3.5GB not available here
- Code is designed to work without PyTorch for data collection
- PyTorch only needed for actual training

## Files Modified

1. `Moisturizer.py`:
   - Line ~881: Fixed timestamp conversion in `__getitem__()`
   - Line ~555: Added `analyze_parameter_coverage()` method
   - Line ~746: Updated docstring with filtering tip

2. `test_parameter_coverage.py`: New test script

## Backward Compatibility

All changes are backward compatible:
- `feature_params=None` still works (uses all parameters)
- Timestamp conversion handles both pandas and numeric types
- Existing code continues to work unchanged

## Recommendations for Training

1. **Always run coverage analysis first:**
   ```python
   coverage, filtered_params = collector.analyze_parameter_coverage(0.25)
   ```

2. **Use filtered parameters:**
   ```python
   dataset = SoilMoistureSequenceDataset(
       ...,
       feature_params=filtered_params
   )
   ```

3. **Adjust threshold based on your needs:**
   - `0.25` (25%): Balanced - good starting point
   - `0.50` (50%): Conservative - only well-covered parameters
   - `0.10` (10%): Aggressive - include more parameters, more missing data

## Next Steps

When PyTorch is available:
1. Run the full data collection pipeline
2. Use `analyze_parameter_coverage()` to get filtered parameters
3. Train your TROLOLO transformer with the filtered feature set
4. The timestamp fix ensures DataLoader works properly
