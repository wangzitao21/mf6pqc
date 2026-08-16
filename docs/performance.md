# Performance and profiling policy

MF6PQC performance is usually dominated by repeated native MODFLOW and
PhreeqcRM solves, so optimization must begin with measured call counts rather
than Python micro-optimisation.

## Optimisations already enforced

- MODFLOW addresses and live NumPy pointers are cached once per run.
- Component slices and concentration buffers are allocated once and reused.
- TDIS steps are expanded once, including stable geometric schedules.
- SIA difference buffers are reused; source tolerance checks do not allocate a
  full temporary array on every iteration.
- Solver IDs and MXITER pointers are cached rather than rediscovered each step.
- Selected output is retained only at requested save steps.
- Progress printing is rate-limited.

## Measurement protocol

For a performance change, record at minimum:

```text
case, commit, Python, MODFLOW, PhreeqcRM, CPU, nthreads,
nxyz, logical steps, chemistry calls, MODFLOW solve calls,
SIA iterations, saved frames, wall seconds
```

Compare the same numerical output and convergence tolerances. A faster result
that changes time discretisation, chemistry calls, or convergence policy is a
scientific-method change, not a pure optimization.

## Highest-value future work

1. Add per-kernel timers around MODFLOW, PhreeqcRM, feedback, and serialization.
2. Add optional streamed result storage for very large snapshot sets.
3. Investigate chemistry load balancing and thread scaling case by case.
4. Reduce SIA iteration counts through justified relaxation/acceleration, with
   convergence and mass-balance tests.
5. Profile long Xie2015 cases one at a time before changing array layout or
   native-call frequency.

Avoid JIT or GPU work until profiling shows Python calculations are material.
Native solver calls, data transfer, and nonlinear iteration count are more
likely to control runtime.

