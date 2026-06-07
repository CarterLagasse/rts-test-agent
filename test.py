"""
High-Performance Heuristic Agent for MicroRTS-Py  (v2 - fixed)
===============================================================

Observation space: (num_envs, H, W, 29) one-hot planes
  [0:5]   HP         (0,1,2,3,>=4)
  [5:10]  Resources  (0,1,2,3,>=4)
  [10:13] Owner      (none, p1, p2)
  [13:21] UnitType   (none,resource,base,barrack,worker,light,heavy,ranged)
  [21:27] CurAction  (none,move,harvest,return,produce,attack)
  [27:29] Terrain    (free, wall)

Action mask per cell: 78 bits split as:
  [0:6]   action type  (NOOP,move,harvest,return,produce,attack)
  [6:10]  move dir     (N,E,S,W)
  [10:14] harvest dir  (N,E,S,W)
  [14:18] return dir   (N,E,S,W)
  [18:22] produce dir  (N,E,S,W)
  [22:29] produce type (resource,base,barrack,worker,light,heavy,ranged)
  [29:78] attack target (7x7 relative grid, flat-indexed)

Output action vector per cell (7 values):
  [0] action_type   [1] move_dir   [2] harvest_dir   [3] return_dir
  [4] produce_dir   [5] produce_type   [6] attack_flat_idx

KEY RULES (discovered by studying the codebase):
  - BASES produce workers
  - WORKERS build barracks (by producing into adjacent empty cell)
  - BARRACKS produce combat units (light, heavy, ranged)
  - Units currently mid-action have only NOOP valid in the mask
  - The env's source_unit_mask filters which cells actually execute actions

STRATEGY:
  Phase 1 (always): Base produces workers; one worker harvests, one builds barracks
  Phase 2: Barracks produce light+ranged mix; workers continue harvesting
  Phase 3: All combat units attack-move toward nearest enemy
  Emergency: If enemies reach our base, workers join the attack
"""

import numpy as np

# ──────────────────────────────────────────────────────────────
# Observation plane offsets
# ──────────────────────────────────────────────────────────────
HP_START      = 0
RES_START     = 5
OWNER_START   = 10
UTYPE_START   = 13
ACTION_START  = 21
TERRAIN_START = 27

OWNER_NONE, OWNER_P1, OWNER_P2 = 0, 1, 2

# Unit type indices (relative to UTYPE_START)
UT_NONE, UT_RESOURCE, UT_BASE, UT_BARRACK = 0, 1, 2, 3
UT_WORKER, UT_LIGHT, UT_HEAVY, UT_RANGED  = 4, 5, 6, 7

# Action type indices
ACT_NOOP, ACT_MOVE, ACT_HARVEST, ACT_RETURN, ACT_PRODUCE, ACT_ATTACK = 0,1,2,3,4,5

# Direction constants
DIR_N, DIR_E, DIR_S, DIR_W = 0, 1, 2, 3

# Produce-type indices in mask [22:29]
PT_RESOURCE, PT_BASE, PT_BARRACK, PT_WORKER = 0, 1, 2, 3
PT_LIGHT, PT_HEAVY, PT_RANGED               = 4, 5, 6

# Mask slices
MASK_ACT   = slice(0,  6)
MASK_MOVE  = slice(6,  10)
MASK_HARV  = slice(10, 14)
MASK_RET   = slice(14, 18)
MASK_PROD  = slice(18, 22)
MASK_PTYPE = slice(22, 29)
MASK_ATK   = slice(29, 78)

ATK_SIZE   = 7
ATK_CENTER = 3

# ──────────────────────────────────────────────────────────────
# Cell-level helpers
# ──────────────────────────────────────────────────────────────

def _argmax_block(cell, start, length):
    return int(cell[start: start + length].argmax())

def _owner(cell):  return _argmax_block(cell, OWNER_START, 3)
def _utype(cell):  return _argmax_block(cell, UTYPE_START, 8)
def _res(cell):    return _argmax_block(cell, RES_START,   5)  # bucket 0-4

def _l1(r1, c1, r2, c2):
    return abs(r1 - r2) + abs(c1 - c2)

def _dir_toward(r_src, c_src, r_dst, c_dst):
    dr, dc = r_dst - r_src, c_dst - c_src
    if abs(dr) >= abs(dc):
        return DIR_S if dr > 0 else DIR_N
    return DIR_E if dc > 0 else DIR_W

def _first_valid_dir(mask4, preferred):
    if mask4[preferred]:
        return preferred
    for d in range(4):
        if mask4[d]:
            return d
    return 0

def _noop():
    return np.array([ACT_NOOP, 0, 0, 0, 0, 0, 0], dtype=np.int32)

def _make_move(d):
    return np.array([ACT_MOVE, d, 0, 0, 0, 0, 0], dtype=np.int32)

def _make_harvest(d):
    return np.array([ACT_HARVEST, 0, d, 0, 0, 0, 0], dtype=np.int32)

def _make_return(d):
    return np.array([ACT_RETURN, 0, 0, d, 0, 0, 0], dtype=np.int32)

def _make_produce(d, ptype):
    return np.array([ACT_PRODUCE, 0, 0, 0, d, ptype, 0], dtype=np.int32)

def _make_attack(flat_idx):
    return np.array([ACT_ATTACK, 0, 0, 0, 0, 0, flat_idx], dtype=np.int32)

# ──────────────────────────────────────────────────────────────
# Navigation helpers
# ──────────────────────────────────────────────────────────────

def _move_toward(r, c, tr, tc, move_mask):
    """Return move action toward target, or None if can't move."""
    if not move_mask.any():
        return None
    pref = _dir_toward(r, c, tr, tc)
    d = _first_valid_dir(move_mask, pref)
    return _make_move(d)

def _best_attack(atk_mask, r, c, enemy_positions):
    """
    Pick the best attack target from the 7x7 flat grid.
    Prefers low-HP enemies (harv/worker > base > combat), else first valid bit.
    """
    best_flat = -1
    best_dist = 9999
    for (er, ec) in enemy_positions:
        dr = er - r + ATK_CENTER
        dc = ec - c + ATK_CENTER
        if 0 <= dr < ATK_SIZE and 0 <= dc < ATK_SIZE:
            flat = dr * ATK_SIZE + dc
            if atk_mask[flat]:
                dist = _l1(r, c, er, ec)
                if dist < best_dist:
                    best_dist = dist
                    best_flat = flat
    if best_flat >= 0:
        return best_flat
    # fallback: first valid bit
    valid = np.where(atk_mask)[0]
    return int(valid[0]) if len(valid) else 0

def _attack_or_move(r, c, mask78, enemy_positions):
    """Try attack first, then move toward nearest enemy."""
    atk_mask  = mask78[MASK_ATK]
    move_mask = mask78[MASK_MOVE]
    act_mask  = mask78[MASK_ACT]

    if act_mask[ACT_ATTACK] and atk_mask.any() and enemy_positions:
        flat = _best_attack(atk_mask, r, c, enemy_positions)
        return _make_attack(flat)

    if act_mask[ACT_MOVE] and move_mask.any() and enemy_positions:
        nearest = min(enemy_positions, key=lambda ep: _l1(r, c, ep[0], ep[1]))
        mv = _move_toward(r, c, nearest[0], nearest[1], move_mask)
        if mv is not None:
            return mv

    return None

# ──────────────────────────────────────────────────────────────
# Environment parser
# ──────────────────────────────────────────────────────────────

def _parse_env(obs):
    """Parse one env's (H, W, 29) observation into a summary dict."""
    H, W = obs.shape[:2]
    p1_units  = []   # (r, c, utype, cell_idx)
    p2_units  = []   # (r, c, utype, cell_idx)
    resources = []   # (r, c)

    for r in range(H):
        for c in range(W):
            cell  = obs[r, c]
            owner = _owner(cell)
            utype = _utype(cell)
            idx   = r * W + c
            if owner == OWNER_P1:
                p1_units.append((r, c, utype, idx))
            elif owner == OWNER_P2:
                p2_units.append((r, c, utype, idx))
            elif utype == UT_RESOURCE:
                resources.append((r, c))

    return {
        "H": H, "W": W,
        "p1_units":  p1_units,
        "p2_units":  p2_units,
        "resources": resources,
        "n_workers":  sum(1 for u in p1_units if u[2] == UT_WORKER),
        "n_barracks": sum(1 for u in p1_units if u[2] == UT_BARRACK),
        "n_bases":    sum(1 for u in p1_units if u[2] == UT_BASE),
        "n_light":    sum(1 for u in p1_units if u[2] == UT_LIGHT),
        "n_heavy":    sum(1 for u in p1_units if u[2] == UT_HEAVY),
        "n_ranged":   sum(1 for u in p1_units if u[2] == UT_RANGED),
        "n_combat":   sum(1 for u in p1_units if u[2] in (UT_LIGHT, UT_HEAVY, UT_RANGED)),
        "enemy_pos":  [(r, c) for (r, c, _, _) in p2_units],
    }

# ──────────────────────────────────────────────────────────────
# Per-unit decision logic
# ──────────────────────────────────────────────────────────────

def _decide(r, c, cell, mask78, env_info, worker_role):
    """
    worker_role: 'harvest', 'build', or 'attack'
    Returns a 7-element action array.
    """
    act_mask   = mask78[MASK_ACT]
    move_mask  = mask78[MASK_MOVE]
    harv_mask  = mask78[MASK_HARV]
    ret_mask   = mask78[MASK_RET]
    prod_mask  = mask78[MASK_PROD]
    ptype_mask = mask78[MASK_PTYPE]
    atk_mask   = mask78[MASK_ATK]

    utype      = _utype(cell)
    p1_units   = env_info["p1_units"]
    p2_units   = env_info["p2_units"]
    resources  = env_info["resources"]
    enemy_pos  = env_info["enemy_pos"]
    n_workers  = env_info["n_workers"]
    n_barracks = env_info["n_barracks"]
    n_combat   = env_info["n_combat"]

    bases   = [(br, bc) for (br, bc, bt, _) in p1_units if bt == UT_BASE]

    # ── No non-NOOP actions available → NOOP ─────────────────
    if not act_mask[1:].any():
        return _noop()

    # ════════════════════════════════════════════════════════════
    # BASE: produce workers
    # ════════════════════════════════════════════════════════════
    if utype == UT_BASE:
        if act_mask[ACT_PRODUCE] and prod_mask.any() and ptype_mask[PT_WORKER]:
            # Spawn away from nearest enemy
            pref = DIR_S
            if enemy_pos:
                nearest_e = min(enemy_pos, key=lambda ep: _l1(r, c, ep[0], ep[1]))
                # opposite direction from enemy
                opp = _dir_toward(nearest_e[0], nearest_e[1], r, c)
                if prod_mask[opp]:
                    pref = opp
            d = _first_valid_dir(prod_mask, pref)
            return _make_produce(d, PT_WORKER)
        return _noop()

    # ════════════════════════════════════════════════════════════
    # BARRACK: produce combat units (light / ranged mix)
    # ════════════════════════════════════════════════════════════
    if utype == UT_BARRACK:
        if act_mask[ACT_PRODUCE] and prod_mask.any():
            # Prefer ranged, then light, then heavy
            n_ranged = env_info["n_ranged"]
            n_light  = env_info["n_light"]
            choice   = None
            if ptype_mask[PT_RANGED] and n_ranged <= n_light:
                choice = PT_RANGED
            elif ptype_mask[PT_LIGHT]:
                choice = PT_LIGHT
            elif ptype_mask[PT_HEAVY]:
                choice = PT_HEAVY
            elif ptype_mask[PT_RANGED]:
                choice = PT_RANGED

            if choice is not None:
                # Produce toward nearest enemy (they'll come out facing the fight)
                pref = DIR_S
                if enemy_pos:
                    nearest_e = min(enemy_pos, key=lambda ep: _l1(r, c, ep[0], ep[1]))
                    pref = _dir_toward(r, c, nearest_e[0], nearest_e[1])
                d = _first_valid_dir(prod_mask, pref)
                return _make_produce(d, choice)
        return _noop()

    # ════════════════════════════════════════════════════════════
    # WORKER logic
    # ════════════════════════════════════════════════════════════
    if utype == UT_WORKER:
        carrying = _res(cell) > 0

        # ── If carrying resources, return to base ─────────────
        if carrying:
            if act_mask[ACT_RETURN] and ret_mask.any():
                pref = DIR_N
                if bases:
                    nb = min(bases, key=lambda b: _l1(r, c, b[0], b[1]))
                    pref = _dir_toward(r, c, nb[0], nb[1])
                d = _first_valid_dir(ret_mask, pref)
                return _make_return(d)
            # Can't return directly — move toward base
            if bases and act_mask[ACT_MOVE] and move_mask.any():
                nb = min(bases, key=lambda b: _l1(r, c, b[0], b[1]))
                mv = _move_toward(r, c, nb[0], nb[1], move_mask)
                if mv is not None:
                    return mv
            return _noop()

        # ── Worker role: BUILD (if no barracks) ───────────────
        if worker_role == 'build' or (n_barracks == 0 and act_mask[ACT_PRODUCE] and prod_mask.any() and ptype_mask[PT_BARRACK]):
            if act_mask[ACT_PRODUCE] and prod_mask.any() and ptype_mask[PT_BARRACK]:
                # Build barracks in any free adjacent cell
                # Prefer direction away from enemies and toward center
                H, W = env_info["H"], env_info["W"]
                pref = _dir_toward(r, c, H // 2, W // 2)
                d = _first_valid_dir(prod_mask, pref)
                return _make_produce(d, PT_BARRACK)
            # Can't build yet — move to find a free spot, or harvest for now
            # Fall through to harvest logic

        # ── Worker role: ATTACK (emergency / rush) ────────────
        if worker_role == 'attack' and enemy_pos:
            atk = _attack_or_move(r, c, mask78, enemy_pos)
            if atk is not None:
                return atk

        # ── Default: HARVEST ──────────────────────────────────
        if resources:
            nearest_res = min(resources, key=lambda res: _l1(r, c, res[0], res[1]))
            dist = _l1(r, c, nearest_res[0], nearest_res[1])

            if dist == 1 and act_mask[ACT_HARVEST] and harv_mask.any():
                pref = _dir_toward(r, c, nearest_res[0], nearest_res[1])
                d = _first_valid_dir(harv_mask, pref)
                return _make_harvest(d)

            if act_mask[ACT_MOVE] and move_mask.any():
                mv = _move_toward(r, c, nearest_res[0], nearest_res[1], move_mask)
                if mv is not None:
                    return mv

        # No resources — if enemies exist, attack-move as last resort
        if enemy_pos:
            atk = _attack_or_move(r, c, mask78, enemy_pos)
            if atk is not None:
                return atk

        return _noop()

    # ════════════════════════════════════════════════════════════
    # COMBAT UNITS (light, heavy, ranged)
    # ════════════════════════════════════════════════════════════
    if utype in (UT_LIGHT, UT_HEAVY, UT_RANGED):
        atk = _attack_or_move(r, c, mask78, enemy_pos)
        if atk is not None:
            return atk
        return _noop()

    # Everything else (resources, etc.) → NOOP
    return _noop()


# ──────────────────────────────────────────────────────────────
# Worker role assignment
# ──────────────────────────────────────────────────────────────

def _assign_worker_roles(env_info):
    """
    Returns a dict mapping (r, c) -> role string for workers.

    Roles:
      'build'   — designated builder (builds barracks)
      'harvest' — primary resource gatherer
      'attack'  — worker rush or emergency defense
    """
    workers = [(r, c, idx) for (r, c, t, idx) in env_info["p1_units"] if t == UT_WORKER]
    n_barracks = env_info["n_barracks"]
    n_combat   = env_info["n_combat"]
    enemy_pos  = env_info["enemy_pos"]
    resources  = env_info["resources"]
    roles = {}

    if not workers:
        return roles

    # Sort workers by cell index for stable assignment
    workers_sorted = sorted(workers, key=lambda w: w[2])

    if n_barracks == 0:
        # First worker builds, rest harvest
        roles[(workers_sorted[0][0], workers_sorted[0][1])] = 'build'
        for w in workers_sorted[1:]:
            roles[(w[0], w[1])] = 'harvest'
    elif not resources:
        # No resources to harvest — all attack-move
        for w in workers_sorted:
            roles[(w[0], w[1])] = 'attack'
    elif n_combat == 0 and enemy_pos and len(workers_sorted) >= 2:
        # No military at all — rush with spare workers, keep one harvesting
        roles[(workers_sorted[0][0], workers_sorted[0][1])] = 'harvest'
        for w in workers_sorted[1:]:
            roles[(w[0], w[1])] = 'attack'
    else:
        # Normal: all workers harvest
        for w in workers_sorted:
            roles[(w[0], w[1])] = 'harvest'

    return roles


# ──────────────────────────────────────────────────────────────
# Main policy
# ──────────────────────────────────────────────────────────────

def policy(observation: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    observation : np.ndarray  shape (num_envs, H, W, 29)
    action_mask : np.ndarray  shape (num_envs, H*W, 78)

    Returns
    -------
    actions : np.ndarray  shape (num_envs, H*W, 7)
    """
    num_envs = observation.shape[0]
    H, W     = observation.shape[1], observation.shape[2]
    map_size = H * W

    actions = np.zeros((num_envs, map_size, 7), dtype=np.int32)

    for env_idx in range(num_envs):
        obs_e    = observation[env_idx]   # (H, W, 29)
        mask_e   = action_mask[env_idx]   # (H*W, 78)

        env_info     = _parse_env(obs_e)
        worker_roles = _assign_worker_roles(env_info)

        for cell_idx in range(map_size):
            r = cell_idx // W
            c = cell_idx  % W
            cell   = obs_e[r, c]
            mask78 = mask_e[cell_idx]

            # Skip cells not owned by player 1
            if _owner(cell) != OWNER_P1:
                continue

            act_mask = mask78[MASK_ACT]
            if not act_mask.any():
                continue

            utype = _utype(cell)
            role  = worker_roles.get((r, c), 'harvest') if utype == UT_WORKER else 'n/a'

            unit_action = _decide(r, c, cell, mask78, env_info, role)

            # Safety: ensure chosen action_type is valid in mask
            chosen_at = unit_action[0]
            if not act_mask[chosen_at]:
                valid_ats = np.where(act_mask)[0]
                unit_action = _noop()
                if len(valid_ats) and valid_ats[0] != ACT_NOOP:
                    unit_action[0] = int(valid_ats[0])

            actions[env_idx, cell_idx] = unit_action

    return actions
