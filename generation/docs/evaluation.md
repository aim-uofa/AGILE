# Generated-mesh evaluation

Use this optional evaluator when a reference mesh is available. It measures object geometry separately from the [hand-object reconstruction metrics](../../optimization/README.md#evaluation).

```bash
python -m agile3d.benchmark_metrics /ABS/generated.glb /ABS/reference.obj \
  --ground-truth-unit-in-meters 1 --output /ABS/mesh-metrics.json
```

Set the reference conversion to `1` for meters or `0.001` for millimeters. Add `--generated-unit-in-meters` only when the generated asset's metric units are known independently.

## Metrics

The evaluator samples mesh surfaces and reports:

- Unsquared Chamfer distance in meters and squared Chamfer distance in m², each using half the sum of the two directional means.
- F-scores at 5, 10 and 20 mm.
- Geometry validity, connected components, extents and volume.

Similarity registration is reported for all valid meshes. Rigid registration is also reported when the generated units are supplied. The output records registration settings, sample counts, seeds and file hashes. Registration and scoring use separate samples.

These values are not interchangeable with reconstruction `cd_right`, which uses a different alignment and distance convention. Compare generated assets with the same units, sampling and registration settings. Report failed assets alongside successful evaluations; agent review scores are not geometric metrics.
