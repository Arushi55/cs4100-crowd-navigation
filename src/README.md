# Crowd Navigation

## Pedestrian Behavior Changelog

1. Obstacle-aware spawn and goal sampling.
Before: Pedestrians could spawn or target points inside blocked geometry.
After: Spawn and goal points are sampled with obstacle clearance and fallback scans.

2. Behavior mix scaling with `--pedestrians`.
Before: Scenario behavior ratios drifted when total pedestrian count changed.
After: Behavior counts are proportionally scaled to preserve intended mix.

3. Scenario mix rebalancing for `home` and `shopping_center`.
Before: Too many low-mobility or oscillatory agents in constrained layouts.
After: Mix favors smoother moving behaviors with fewer pathological agents.

4. Stationary behavior mobility increase.
Before: Stationary agents almost never moved.
After: Movement probability increased so they still exhibit occasional intent.

5. Simulation unit calibration layer.
Before: Speeds were pixel-step values without explicit physical conversion.
After: Shared m/s <-> px/step conversions and fixed simulation timestep are defined.

6. Pedestrian dynamics limits.
Before: Direction and speed could change too abruptly frame to frame.
After: Max acceleration, max turn rate, and speed caps are enforced.

7. TTC anticipatory avoidance.
Before: Avoidance was mostly reactive and could produce late jitter.
After: Time-to-collision side-step force adds earlier smoother evasive motion.

8. Zigzag behavior smoothing.
Before: Zigzag heading changes were abrupt resets.
After: Zigzag offsets drift smoothly around goal heading with velocity blending.

9. Movement smoothing and stuck recovery tuning.
Before: Strong reversals and kicks could cause harsh oscillations.
After: Velocity smoothing and gentler, cooldown-gated recovery improve stability.

10. Goal-hit lifecycle without instant teleport.
Before: Reached pedestrians were teleported to new random spawn points.
After: Pedestrians dwell briefly at goal, then continue with a new destination.

11. Region-specific dwell and scenario-specific flow.
Before: All scenarios used one dwell pattern and homogeneous lifecycle behavior.
After: Dwell depends on destination region, and each scenario has its own flow ratio.

12. Markov OD transition for reassignment.
Before: Next goals were mostly random.
After: Next destination region is sampled from scenario transition weights.

13. Initial behavior realism pass.
Before: Initial route and velocity were simplistic and often static at t=0.
After: Initial OD-biased routing, profile heterogeneity, and seeded motion are applied.

14. Lifecycle logic deduplication.
Before: `main.py` and `crowd_env.py` had duplicated reassignment and flow logic.
After: Shared lifecycle module keeps behavior consistent and reduces maintenance risk.
