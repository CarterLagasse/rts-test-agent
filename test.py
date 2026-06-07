"""
High-Performance Heuristic Agent for MicroRTS-Py
=================================================
Observation space: (num_envs, H, W, 29) one-hot planes
  [0:5]   HP         (0,1,2,3,>=4)
  [5:10]  Resources  (0,1,2,3,>=4)
  [10:13] Owner      (none, p1, p2)
  [13:21] UnitType   (none,resource,base,barrack,worker,light,heavy,ranged)
  [21:27] CurAction  (none,move,harvest,return,produce,attack)
  [27:29] Terrain    (free, wall)

Action mask per cell: 78 bits
  [0:6]   action type  (NOOP,move,harvest,return,produce,attack)
  [6:10]  move dir     (N,E,S,W)
  [10:14] harvest dir  (N,E,S,W)
  [14:18] return dir   (N,E,S,W)
  [18:22] produce dir  (N,E,S,W)
  [22:29] produce type (resource,base,barrack,worker,light,heavy,ranged)
  [29:78] attack target (7x7 relative grid)

Strategy:
  Phase 1 – Economy bootstrap
    * Worker → harvest mineral → return to base (loop)
    * Base produces a 2nd worker when resources allow
  Phase 2 – Military build-up  (resources >= 2)
    * Build barracks if none exist
    * Barracks produce light units
  Phase 3 – Attack
    * Combat units (light/heavy/ranged) seek the nearest enemy
    * Idle workers without harvest duty attack-move toward enemies
  Special:
    * Ranged units prefer to attack-move from distance
    * If we have barracks and enough resources, produce heavy/ranged mix
    * Unused bases occasionally produce more workers
"""

import numpy as np
from typing import Tuple

# ──────────────────────────────────────────────────────────────
# Observation plane offsets (one-hot)
# ──────────────────────────────────────────────────────────────
HP_START       = 0
RES_START      = 5
OWNER_START    = 10
UTYPE_START    = 13
ACTION_START   = 21
TERRAIN_START  = 27

# Owner indices (relative to OWNER_START)
OWNER_NONE  = 0
OWNER_P1    = 1
OWNER_P2    = 2

# Unit type indices (relative to UTYPE_START)
UT_NONE     = 0
UT_RESOURCE = 1
UT_BASE     = 2
UT_BARRACK  = 3
UT_WORKER   = 4
UT_LIGHT    = 5
UT_HEAVY    = 6
UT_RANGED   = 7

# Current-action indices (relative to ACTION_START)
CA_NONE     = 0
CA_MOVE     = 1
CA_HARVEST  = 2
CA_RETURN   = 3
CA_PRODUCE  = 4
CA_ATTACK   = 5

# Action-type indices (in the output action vector)
ACT_NOOP    = 0
ACT_MOVE    = 1
ACT_HARVEST = 2
ACT_RETURN  = 3
ACT_PRODUCE = 4
ACT_ATTACK  = 5

# Direction constants (N=0, E=1, S=2, W=3)
DIR_N, DIR_E, DIR_S, DIR_W = 0, 1, 2, 3

# Produce-type constants (index into [22:29] of mask)
PT_RESOURCE = 0
PT_BASE     = 1
PT_BARRACK  = 2
PT_WORKER   = 3
PT_LIGHT    = 4
PT_HEAVY    = 5
PT_RANGED   = 6

# Direction deltas: (dy, dx)  [row, col] = [y, x]
DIR_DELTA = {
    DIR_N: (-1,  0),
    DIR_E: ( 0,  1),
    DIR_S: ( 1,  0),
    DIR_W: ( 0, -1),
}

# ──────────────────────────────────────────────────────────────
# Mask group slices (within the 78-bit action mask per cell)
# ──────────────────────────────────────────────────────────────
MASK_ACT   = slice(0,  6)
MASK_MOVE  = slice(6,  10)
MASK_HARV  = slice(10, 14)
MASK_RET   = slice(14, 18)
MASK_PROD  = slice(18, 22)
MASK_PTYPE = slice(22, 29)
MASK_ATK   = slice(29, 78)

ATK_SIZE   = 7  # 7x7 attack grid
ATK_CENTER = 3  # center index of 7x7


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _plane(obs_cell: np.ndarray, start: int) -> int:
    """Return the active index within a one-hot block starting at `start`."""
    # obs_cell shape: (29,)
    block = obs_cell[start: start + 8]  # generous upper bound
    idx = block.argmax()
    return int(idx)


def _owner(obs_cell: np.ndarray) -> int:
    return _plane(obs_cell, OWNER_START)


def _utype(obs_cell: np.ndarray) -> int:
    return _plane(obs_cell, UTYPE_START)


def _cur_action(obs_cell: np.ndarray) -> int:
    return _plane(obs_cell, ACTION_START)


def _resources(obs_cell: np.ndarray) -> int:
    """Estimated resource count (returns bucket index 0-4)."""
    return int(obs_cell[RES_START: RES_START + 5].argmax())


def _hp(obs_cell: np.ndarray) -> int:
    return int(obs_cell[HP_START: HP_START + 5].argmax())


def _is_wall(obs_cell: np.ndarray) -> bool:
    return bool(obs_cell[TERRAIN_START + 1] == 1)


def _l1(r1, c1, r2, c2) -> int:
    return abs(r1 - r2) + abs(c1 - c2)


def _dir_toward(r_src, c_src, r_dst, c_dst) -> int:
    """Cardinal direction from (r_src,c_src) toward (r_dst,c_dst)."""
    dr = r_dst - r_src
    dc = c_dst - c_src
    if abs(dr) >= abs(dc):
        return DIR_S if dr > 0 else DIR_N
    else:
        return DIR_E if dc > 0 else DIR_W


def _best_valid_dir(mask_4: np.ndarray, preferred: int) -> int:
    """Return preferred direction if valid, else any valid direction."""
    if mask_4[preferred]:
        return preferred
    for d in range(4):
        if mask_4[d]:
            return d
    return 0  # fallback (shouldn't happen if action type is valid)


def _best_attack_target(atk_mask: np.ndarray, r_src, c_src,
                         enemy_positions, H, W) -> int:
    """
    Choose the best attack target index (0-48) from the 7x7 relative grid.
    Prefers enemies that are in the mask; falls back to any valid bit.
    """
    # Map enemy positions to relative attack offsets
    best_idx = -1
    best_priority = -1
    for (er, ec) in enemy_positions:
        dr = er - r_src + ATK_CENTER
        dc = ec - c_src + ATK_CENTER
        if 0 <= dr < ATK_SIZE and 0 <= dc < ATK_SIZE:
            flat = dr * ATK_SIZE + dc
            if atk_mask[flat]:
                priority = 0  # prefer weaker enemies (lower HP bucket handled by unit type)
                if priority > best_priority:
                    best_priority = priority
                    best_idx = flat
    if best_idx >= 0:
        return best_idx
    # fallback: first valid bit
    for i in range(ATK_SIZE * ATK_SIZE):
        if atk_mask[i]:
            return i
    return 0


def _masked_softmax_sample(logits_or_mask: np.ndarray) -> int:
    """Sample from valid (1) entries proportionally (uniform among valid)."""
    valid = np.where(logits_or_mask > 0)[0]
    if len(valid) == 0:
        return 0
    return int(np.random.choice(valid))


# ──────────────────────────────────────────────────────────────
# Per-environment state tracking (lightweight, reset each call
# since we can reconstruct everything from obs)
# ──────────────────────────────────────────────────────────────

def _parse_env(obs: np.ndarray) -> dict:
    """
    Parse one environment's observation (H, W, 29) into useful structures.
    Returns:
      - p1_units:  list of (r,c,utype) for player 1
      - p2_units:  list of (r,c,utype) for player 2
      - resources: list of (r,c) for mineral nodes
      - p1_resources_on_map: estimated total resources carried by p1 units
    """
    H, W = obs.shape[:2]
    p1_units  = []
    p2_units  = []
    resources = []

    for r in range(H):
        for c in range(W):
            cell = obs[r, c]
            owner = _owner(cell)
            utype = _utype(cell)
            if owner == OWNER_P1:
                p1_units.append((r, c, utype))
            elif owner == OWNER_P2:
                p2_units.append((r, c, utype))
            elif utype == UT_RESOURCE:
                resources.append((r, c))

    return {
        "p1_units":  p1_units,
        "p2_units":  p2_units,
        "resources": resources,
        "H": H,
        "W": W,
    }


# ──────────────────────────────────────────────────────────────
# Core decision function for a single unit in one environment
# ──────────────────────────────────────────────────────────────

def _decide_unit(
    r: int, c: int,
    cell: np.ndarray,
    mask78: np.ndarray,
    env_info: dict,
) -> np.ndarray:
    """
    Returns a 7-element action vector for the unit at (r,c).
    Elements: [action_type, move_dir, harvest_dir, return_dir,
               produce_dir, produce_type, attack_flat_idx]
    """
    act_mask  = mask78[MASK_ACT]
    move_mask = mask78[MASK_MOVE]
    harv_mask = mask78[MASK_HARV]
    ret_mask  = mask78[MASK_RET]
    prod_mask = mask78[MASK_PROD]
    ptype_mask = mask78[MASK_PTYPE]
    atk_mask  = mask78[MASK_ATK]

    utype    = _utype(cell)
    cur_act  = _cur_action(cell)
    H, W     = env_info["H"], env_info["W"]
    p1_units = env_info["p1_units"]
    p2_units = env_info["p2_units"]
    resources= env_info["resources"]

    # Counts
    n_workers  = sum(1 for (_, _, t) in p1_units if t == UT_WORKER)
    n_barracks = sum(1 for (_, _, t) in p1_units if t == UT_BARRACK)
    n_combat   = sum(1 for (_, _, t) in p1_units if t in (UT_LIGHT, UT_HEAVY, UT_RANGED))
    n_bases    = sum(1 for (_, _, t) in p1_units if t == UT_BASE)

    enemy_positions = [(er, ec) for (er, ec, _) in p2_units]

    # Default action vector
    action = np.array([ACT_NOOP, 0, 0, 0, 0, PT_WORKER, 0], dtype=np.int32)

    # ── Helper: move toward target ──────────────────────────────
    def move_toward(tr, tc):
        if not act_mask[ACT_MOVE]:
            return None
        pref = _dir_toward(r, c, tr, tc)
        d = _best_valid_dir(move_mask, pref)
        return np.array([ACT_MOVE, d, 0, 0, 0, 0, 0], dtype=np.int32)

    def attack_nearest():
        """Try to attack; if can't, move toward nearest enemy."""
        if act_mask[ACT_ATTACK] and atk_mask.any():
            tgt = _best_attack_target(atk_mask, r, c, enemy_positions, H, W)
            return np.array([ACT_ATTACK, 0, 0, 0, 0, 0, tgt], dtype=np.int32)
        if enemy_positions and act_mask[ACT_MOVE] and move_mask.any():
            nearest = min(enemy_positions, key=lambda ep: _l1(r, c, ep[0], ep[1]))
            return move_toward(nearest[0], nearest[1])
        return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # WORKER logic
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if utype == UT_WORKER:
        carrying = _resources(cell) > 0
        # If currently mid-harvest or mid-return, let it continue (NOOP for busy)
        # (env handles this; if mask says only NOOP valid, we NOOP)
        if not any(act_mask[1:]):  # only NOOP valid
            return action  # NOOP

        if carrying:
            # Return resources to base
            if act_mask[ACT_RETURN] and ret_mask.any():
                pref = 0
                bases = [(br, bc) for (br, bc, bt) in p1_units if bt == UT_BASE]
                if bases:
                    nearest_base = min(bases, key=lambda b: _l1(r, c, b[0], b[1]))
                    pref = _dir_toward(r, c, nearest_base[0], nearest_base[1])
                d = _best_valid_dir(ret_mask, pref)
                return np.array([ACT_RETURN, 0, 0, d, 0, 0, 0], dtype=np.int32)
            # Can't return directly, move toward base
            bases = [(br, bc) for (br, bc, bt) in p1_units if bt == UT_BASE]
            if bases and act_mask[ACT_MOVE] and move_mask.any():
                nearest_base = min(bases, key=lambda b: _l1(r, c, b[0], b[1]))
                mv = move_toward(nearest_base[0], nearest_base[1])
                if mv is not None:
                    return mv
        else:
            # Harvest minerals
            if resources:
                nearest_res = min(resources, key=lambda res: _l1(r, c, res[0], res[1]))
                dist = _l1(r, c, nearest_res[0], nearest_res[1])
                if dist == 1 and act_mask[ACT_HARVEST] and harv_mask.any():
                    pref = _dir_toward(r, c, nearest_res[0], nearest_res[1])
                    d = _best_valid_dir(harv_mask, pref)
                    return np.array([ACT_HARVEST, 0, d, 0, 0, 0, 0], dtype=np.int32)
                elif act_mask[ACT_MOVE] and move_mask.any():
                    mv = move_toward(nearest_res[0], nearest_res[1])
                    if mv is not None:
                        return mv

        # If we have enemies and no resources to harvest, fight
        if n_combat == 0 and enemy_positions:
            atk = attack_nearest()
            if atk is not None:
                return atk

        return action  # NOOP

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # BASE logic
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if utype == UT_BASE:
        if not act_mask[ACT_PRODUCE] or not prod_mask.any():
            return action  # busy or can't produce

        # Priority: produce workers first if < 2, else keep at ≤3
        if n_workers < 2 and ptype_mask[PT_WORKER]:
            d = _best_valid_dir(prod_mask, DIR_S)
            # prefer direction away from enemies
            if enemy_positions:
                enemy_avg_r = np.mean([ep[0] for ep in enemy_positions])
                enemy_avg_c = np.mean([ep[1] for ep in enemy_positions])
                away_r = r - (enemy_avg_r - r)
                away_c = c - (enemy_avg_c - c)
                pref = _dir_toward(r, c, away_r, away_c)
                if prod_mask[pref]:
                    d = pref
            return np.array([ACT_PRODUCE, d, 0, 0, d, PT_WORKER, 0], dtype=np.int32)

        # If no barracks and enough workers, produce barracks
        if n_barracks == 0 and n_workers >= 2 and ptype_mask[PT_BARRACK]:
            d = _best_valid_dir(prod_mask, DIR_E)
            return np.array([ACT_PRODUCE, d, 0, 0, d, PT_BARRACK, 0], dtype=np.int32)

        # Otherwise produce more workers (up to 3)
        if n_workers < 3 and ptype_mask[PT_WORKER]:
            d = _best_valid_dir(prod_mask, DIR_S)
            return np.array([ACT_PRODUCE, d, 0, 0, d, PT_WORKER, 0], dtype=np.int32)

        return action

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # BARRACKS logic
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if utype == UT_BARRACK:
        if not act_mask[ACT_PRODUCE] or not prod_mask.any():
            return action

        # Produce mix: prefer ranged if possible, then light, then heavy
        # Strategy: keep ranged:light roughly 1:1, fill heavy as needed
        n_ranged = sum(1 for (_, _, t) in p1_units if t == UT_RANGED)
        n_light  = sum(1 for (_, _, t) in p1_units if t == UT_LIGHT)
        n_heavy  = sum(1 for (_, _, t) in p1_units if t == UT_HEAVY)

        unit_choice = PT_LIGHT  # default
        if ptype_mask[PT_RANGED] and n_ranged <= n_light:
            unit_choice = PT_RANGED
        elif ptype_mask[PT_LIGHT]:
            unit_choice = PT_LIGHT
        elif ptype_mask[PT_HEAVY]:
            unit_choice = PT_HEAVY

        if ptype_mask[unit_choice]:
            # Produce toward enemies
            d = DIR_S
            if enemy_positions:
                nearest_enemy = min(enemy_positions, key=lambda ep: _l1(r, c, ep[0], ep[1]))
                pref = _dir_toward(r, c, nearest_enemy[0], nearest_enemy[1])
                d = _best_valid_dir(prod_mask, pref)
            else:
                d = _best_valid_dir(prod_mask, DIR_S)
            return np.array([ACT_PRODUCE, d, 0, 0, d, unit_choice, 0], dtype=np.int32)

        return action

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # COMBAT UNIT logic (light, heavy, ranged)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if utype in (UT_LIGHT, UT_HEAVY, UT_RANGED):
        atk = attack_nearest()
        if atk is not None:
            return atk
        return action  # NOOP if no enemy in range and can't move

    # RESOURCE NODES – always NOOP
    return action


# ──────────────────────────────────────────────────────────────
# Main policy function
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
    num_envs   = observation.shape[0]
    H, W       = observation.shape[1], observation.shape[2]
    map_size   = H * W

    # Output buffer
    actions = np.zeros((num_envs, map_size, 7), dtype=np.int32)

    for env_idx in range(num_envs):
        obs_e    = observation[env_idx]   # (H, W, 29)
        mask_e   = action_mask[env_idx]   # (H*W, 78)
        env_info = _parse_env(obs_e)

        for cell_idx in range(map_size):
            r = cell_idx // W
            c = cell_idx  % W
            cell   = obs_e[r, c]     # (29,)
            mask78 = mask_e[cell_idx] # (78,)

            # Only act if this cell has at least one non-NOOP action valid
            # and is owned by player 1
            owner = _owner(cell)
            if owner != OWNER_P1:
                # NOOP
                actions[env_idx, cell_idx] = [ACT_NOOP, 0, 0, 0, 0, 0, 0]
                continue

            act_mask = mask78[MASK_ACT]
            if not act_mask.any():
                actions[env_idx, cell_idx] = [ACT_NOOP, 0, 0, 0, 0, 0, 0]
                continue

            unit_action = _decide_unit(r, c, cell, mask78, env_info)

            # Safety check: ensure chosen action_type is actually valid in mask.
            # If not, fall back to first valid action type.
            chosen_at = unit_action[0]
            if not act_mask[chosen_at]:
                valid_ats = np.where(act_mask)[0]
                unit_action[0] = int(valid_ats[0]) if len(valid_ats) > 0 else ACT_NOOP

            actions[env_idx, cell_idx] = unit_action

    return actions
