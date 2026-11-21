// Sokoban solver — room-decomposed push-optimal search.
// One program answers a set of 5 Sokoban levels (i001-i005) in a single run.
// Strategy: the generator lays a 1-wide hallway with 1-4 rooms off it (6x6 slots,
// one door each); a box pushed through a door can never come back, so every box
// stays in its room and the level optimum P* is the SUM of the rooms' optima.
// So: cut the hallway out of the floor, flood-fill the rest into rooms, solve
// each room independently with a push-based A* (pushes as cost, min-assignment
// Manhattan distance as an admissible heuristic, dead-square + frozen 2x2
// pruning), then stitch the move sequences together via the hallway.
// Node 22 runs this file directly (erasable TypeScript only, stdlib only).
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Directions in a fixed order (deterministic tie-breaking everywhere).
// [dx, dy, walk letter, push letter]
const DIRS: ReadonlyArray<readonly [number, number, string, string]> = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

// ---------------------------------------------------------------- level ----
type Level = {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
};

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x] as string;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  // The outermost frame is wall whatever it shows: nothing leaves the board.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

// ---------------------------------------------------------- dead squares ---
// live[i] = a box on square i can still be pushed onto some goal (ignoring the
// other boxes). Computed by reverse pulls from the goals: a box at y can be
// pushed to x (a live square) in direction d when the player can stand at y-d.
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue: number[] = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi] as number;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (y < 0 || y >= wall.length) continue;
      const behind = y - d;
      if (behind < 0 || behind >= wall.length) continue;
      if (!wall[y] && !wall[behind] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

// --------------------------------------------------------------- regions ---
// Flood-fill from `from` around walls and occupied squares. Fills `seen` with
// `stamp` and returns the lowest square reached (the region's name).
function flood(
  level: Level,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  const { w, wall } = level;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n] as number;
    if (p < low) low = p;
    const up = p - w;
    const dn = p + w;
    const lf = p - 1;
    const rt = p + 1;
    if (up >= 0 && wall[up] === 0 && occupied[up] === 0 && seen[up] !== stamp) {
      seen[up] = stamp;
      stack[n++] = up;
    }
    if (dn < wall.length && wall[dn] === 0 && occupied[dn] === 0 && seen[dn] !== stamp) {
      seen[dn] = stamp;
      stack[n++] = dn;
    }
    if (lf >= 0 && wall[lf] === 0 && occupied[lf] === 0 && seen[lf] !== stamp) {
      seen[lf] = stamp;
      stack[n++] = lf;
    }
    if (rt < wall.length && wall[rt] === 0 && occupied[rt] === 0 && seen[rt] !== stamp) {
      seen[rt] = stamp;
      stack[n++] = rt;
    }
  }
  return low;
}

// ------------------------------------------------------------ frozen block --
// Whether the box just pushed to `t` sits in a frozen 2x2 block (four squares
// of walls and boxes, with a box among them off its goal): such a box can never
// move again, so the state is dead.
function frozenBlock(
  level: Level,
  occupied: Uint8Array,
  goal: Uint8Array,
  t: number,
): boolean {
  const { w, wall } = level;
  const starts = [t, t - 1, t - w, t - w - 1];
  for (const a of starts) {
    if (a < 0) continue;
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (d >= wall.length) continue;
    if (!(wall[a] || occupied[a])) continue;
    if (!(wall[b] || occupied[b])) continue;
    if (!(wall[c] || occupied[c])) continue;
    if (!(wall[d] || occupied[d])) continue;
    if (
      (occupied[a] && !goal[a]) ||
      (occupied[b] && !goal[b]) ||
      (occupied[c] && !goal[c]) ||
      (occupied[d] && !goal[d])
    ) {
      return true;
    }
  }
  return false;
}

// ------------------------------------------------------------- heuristic ---
// Minimum total Manhattan distance over all box->goal assignments. For n<=6
// boxes this is exact (brute force over permutations); for more, the sum of
// each box's nearest-goal distance (a weaker but still admissible bound).
// One push moves one box one square, so it can lower this sum by at most 1:
// the heuristic never overestimates the pushes left, so A* stays optimal.
function heuristic(boxes: number[], goals: number[], w: number): number {
  const n = boxes.length;
  if (n === 0) return 0;
  const dist = (b: number, g: number): number =>
    Math.abs((b % w) - (g % w)) + Math.abs(Math.floor(b / w) - Math.floor(g / w));
  if (n <= 6) {
    // Exact min assignment by enumerating goal permutations (deterministic).
    const perm: number[] = [];
    for (let i = 0; i < n; i++) perm.push(i);
    let best = Infinity;
    const rec = (k: number, acc: number): void => {
      if (acc >= best) return;
      if (k === n) {
        if (acc < best) best = acc;
        return;
      }
      for (let i = k; i < n; i++) {
        const t = perm[k] as number;
        perm[k] = perm[i] as number;
        perm[i] = t;
        rec(k + 1, acc + dist(boxes[k] as number, goals[perm[k] as number] as number));
        const t2 = perm[k] as number;
        perm[k] = perm[i] as number;
        perm[i] = t2;
      }
    };
    rec(0, 0);
    return best;
  }
  let sum = 0;
  for (const b of boxes) {
    let m = Infinity;
    for (const g of goals) {
      const d = dist(b, g);
      if (d < m) m = d;
    }
    sum += m;
  }
  return sum;
}

// ------------------------------------------------------------ shortest walk --
// Shortest player walk from `from` to `to` around walls and boxes (BFS, the
// DIRS order first on ties, so the walk is deterministic). Lowercase letters.
function walkTo(
  level: Level,
  occupied: Uint8Array,
  from: number,
  to: number,
): string {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how: string[] = new Array(wall.length);
  const queue: number[] = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi] as number;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (q < 0 || q >= wall.length) continue;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p] as number) s = (how[p] as string) + s;
  return s;
}

// ------------------------------------------------------------ push A* -------
// Push-optimal A* over macro-states. A state is the boxes (sorted) plus the
// player's reachable region (named by its lowest square); each expansion is
// one push (cost 1). Returns { pushes, moves } or null when the state budget
// runs out (or the level is unsolvable). Deterministic: fixed direction order,
// heap ties broken by (f, g, insertion order), budget counted in states.
const MAX_STATES = 400_000;

type SolveResult = { pushes: number; moves: string };

function solvePushAStar(level: Level, maxStates: number = MAX_STATES): SolveResult | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b]);
  if (level.boxes.some((b) => !live[b])) return null;
  if (solved(level.boxes)) return { pushes: 0, moves: '' };

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  // A state's key: region-lowest and boxes as digits in base `size`.
  const keyOf = (boxes: number[], low: number): number => {
    let k = low;
    for (const b of boxes) k = k * size + b;
    return k;
  };

  // State storage (parallel arrays).
  const stBoxes: number[][] = [level.boxes.slice()];
  const stPlayer: number[] = [level.player];
  const stG: number[] = [0];
  const stParent: number[] = [-1];
  const stPushFrom: number[] = [-1];
  const stPushTo: number[] = [-1];
  const stPushDir: number[] = [-1]; // index into DIRS
  const index = new Map<number, number>();
  for (const b of level.boxes) occupied[b] = 1;
  const startLow = flood(level, occupied, level.player, seen, ++stamp, stack);
  for (const b of level.boxes) occupied[b] = 0;
  index.set(keyOf(level.boxes, startLow), 0);

  // Binary min-heap on (f, g, seq).
  type HeapEntry = [number, number, number, number]; // f, g, seq, state
  const heap: HeapEntry[] = [];
  const less = (a: HeapEntry, b: HeapEntry): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: HeapEntry): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i] as HeapEntry, heap[p] as HeapEntry)) break;
      const t = heap[i] as HeapEntry;
      heap[i] = heap[p] as HeapEntry;
      heap[p] = t;
      i = p;
    }
  };
  const heapPop = (): HeapEntry => {
    const top = heap[0] as HeapEntry;
    const last = heap.pop() as HeapEntry;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1] as HeapEntry, heap[c] as HeapEntry)) c++;
        if (!less(heap[c] as HeapEntry, heap[i] as HeapEntry)) break;
        const t = heap[i] as HeapEntry;
        heap[i] = heap[c] as HeapEntry;
        heap[c] = t;
        i = c;
      }
    }
    return top;
  };

  let seq = 0;
  heapPush([heuristic(level.boxes, goals, w), 0, seq++, 0]);
  let found = -1;

  while (heap.length > 0) {
    const [, g, , s] = heapPop();
    if (g !== stG[s]) continue; // stale entry (a better path to s was found)
    const boxes = stBoxes[s] as number[];
    if (solved(boxes)) {
      found = s;
      break;
    }
    if (stBoxes.length >= maxStates) break;
    for (const b of boxes) occupied[b] = 1;
    flood(level, occupied, stPlayer[s] as number, seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi] as number;
      for (let di = 0; di < DIRS.length; di++) {
        const [dx, dy] = DIRS[di] as readonly [number, number, string, string];
        const d = dy * w + dx;
        const t = b + d;
        const behind = b - d;
        // The player must stand on `behind` (inside the reached region) and
        // the target must be open floor, live, and not frozen.
        if (behind < 0 || behind >= size || seen[behind] !== here) continue;
        if (t < 0 || t >= size || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(level, occupied, goal, t);
        let key = -1;
        let next: number[] | null = null;
        let low = -1;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((a, x) => a - x);
          low = flood(level, occupied, b, mark, ++stamp, stack);
          key = keyOf(next, low);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || index.has(key)) continue;
        const ng = g + 1;
        const ns = stBoxes.length;
        index.set(key, ns);
        stBoxes.push(next as number[]);
        stPlayer.push(b);
        stG.push(ng);
        stParent.push(s);
        stPushFrom.push(b);
        stPushTo.push(t);
        stPushDir.push(di);
        heapPush([ng + heuristic(next as number[], goals, w), ng, seq++, ns]);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;

  // Reconstruct: walk to each push's square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = stParent[s] as number) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes.slice();
  for (const s of path) {
    const from = stPushFrom[s] as number;
    const to = stPushTo[s] as number;
    const di = stPushDir[s] as number;
    const d = to - from;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(level, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += (DIRS[di] as readonly [number, number, string, string])[3];
    player = from;
    const ni = boxes.indexOf(from);
    boxes[ni] = to;
    boxes.sort((a, b) => a - b);
  }
  return { pushes: path.length, moves };
}

// ------------------------------------------------------------ rooms ---------
// The level's rooms: cut the dead squares of the player's row (the hallway:
// a box there reaches no goal) out of the floor; each remaining floor part
// holding boxes is a room. Null when the level is no clean set of rooms
// (a box in the hallway, a goal cut off, or boxes != goals in a part).
type Room = { boxes: number[]; goals: number[] };

function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (i >= 0 && i < wall.length && !wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const parts: { boxes: number[]; goals: number[] }[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares: number[] = [s];
    part[s] = parts.length;
    for (let i = 0; i < squares.length; i++) {
      const p = squares[i] as number;
      const nb = [p - w, p + w, p - 1, p + 1];
      for (const q of nb) {
        if (q < 0 || q >= wall.length) continue;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = parts.length;
          squares.push(q);
        }
      }
    }
    parts.push({ boxes: [], goals: [] });
  }
  for (const b of boxes) {
    const p = part[b];
    if (p < 0) return null;
    (parts[p] as { boxes: number[]; goals: number[] }).boxes.push(b);
  }
  for (const g of goals) {
    const p = part[g];
    if (p < 0) return null;
    (parts[p] as { boxes: number[]; goals: number[] }).goals.push(g);
  }
  if (parts.some((r) => r.boxes.length !== r.goals.length)) return null;
  // Deterministic room order: by lowest square (scan order).
  const rooms = parts.filter((r) => r.boxes.length > 0);
  rooms.sort((a, b) => Math.min(...a.boxes) - Math.min(...b.boxes));
  for (const r of rooms) {
    r.boxes.sort((a, b) => a - b);
    r.goals.sort((a, b) => a - b);
  }
  return rooms;
}

// Solve room by room: each room's push-optimal solution in turn, the other
// rooms' boxes standing as walls (on their goals once their room is done),
// the player walking on from where the last room left it. Falls back to a
// whole-level search when the level is no clean set of rooms.
function solveLevel(level: Level): SolveResult | null {
  const rooms = roomsOf(level);
  if (rooms === null) return solvePushAStar(level);
  let player = level.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      const blockers = (other as unknown as { done?: boolean }).done
        ? other.goals
        : other.boxes;
      for (const b of blockers) wall[b] = 1;
    }
    const sub: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice(),
      boxes: room.boxes.slice(),
      player,
    };
    const solved = solvePushAStar(sub);
    if (solved === null) return null;
    pushes += solved.pushes;
    moves += solved.moves;
    // Player's square after the room's moves (walk the letters).
    for (const ch of solved.moves) {
      const c = ch.toLowerCase();
      player += c === 'u' ? -level.w : c === 'd' ? level.w : c === 'l' ? -1 : 1;
    }
    (room as unknown as { done: boolean }).done = true;
  }
  return { pushes, moves };
}

// ------------------------------------------------------------------ main ----
function solve(text: string): string {
  const level = parseLevel(text);
  const solved = solveLevel(level);
  return solved === null ? '' : solved.moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
