# Sokoban solver — room-decomposed push-optimal search

[![sokoban-solver-01-run-20261121](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/vivid-raven-88/sokoban-solver-01-run-20261121/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/vivid-raven-88/sokoban-solver-01-run-20261121/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)

A Sokoban solver that answers every instance at its push-optimal count.
The generator lays a 1-wide hallway with 1–4 rooms off it (6×6 slots, one
door each); a box pushed through a door can never come back, so every box
stays in its room and the level optimum is the sum of the rooms' optima.
This solver cuts the hallway out of the floor, flood-fills the rest into
rooms, and solves each room independently with a push-based A* (pushes as
cost, minimum-assignment Manhattan distance as an admissible heuristic,
dead-square and frozen 2×2 deadlock pruning), then stitches the move
sequences together by walking the hallway between rooms.

Entry for the [sokoban-solver-01-run-20261121](https://make.fun/bounty/sokoban-solver-01-run-20261121)
bounty on [make.fun](https://make.fun). The program (`main.ts`, Node 22,
standard library only) reads every level from `/input/<id>/level.txt` and
writes `/output/<id>/solution.txt`. Practice set: all 5 instances solved at
exactly P* (13/16/29/47/60 pushes), 1,000,000 points, in ~0.6 s.
