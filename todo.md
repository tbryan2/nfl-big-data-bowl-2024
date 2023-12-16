# To do

1. Add weighting around the selected target. Eg.
- Clean up code in the PSO class and also the objective function for new target selection code.

```
[0, 0]
[0, 0]
[1, 1]
[0, 0]
[0, 0]

=>

[0, 0]
[0.2, 0.2]
[1, 1]
[0.2, 0.2]
[0, 0]
```

2. Adjust velocity to match the ball carrier's speed (assumption)
3. Increase exponential smoothing on paths (if more iterations does not smooth paths already)
4. Choose and test sample plays