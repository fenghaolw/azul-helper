# MCTS Simulation Performance Analysis Summary

## Key Findings

### Current Performance
- **Simulation time**: ~6.3ms per simulation (not 3.2ms as initially reported)
- **Move selection time**: ~630ms for 100 simulations
- **Simulations per second**: ~158

### Primary Bottleneck Identified
**Random rollouts are extremely long and expensive:**
- Average rollout length: **107.6 moves**
- Average rollout time: **5.8ms**
- Rollout time represents **~90%** of total simulation time

### Detailed Profiling Results

#### From cProfile Analysis (150 simulations):
```
Operation                           Time (ms)  % of Total
===================================================
RandomRolloutEvaluator              1,532      99.3%
get_legal_actions (16,288 calls)      777      50.4%
can_place_tiles_on_pattern_line       348      22.6%
Action hashing/equality               157      10.2%
```

#### Component Breakdown:
- **State copying**: 0.006ms (very fast)
- **Legal actions**: 0.126ms per call
- **State conversion**: 0.119ms
- **Apply action**: 0.079ms
- **Random rollout**: 6.603ms (main bottleneck)

## Root Causes

### 1. Extremely Long Azul Games
- Standard MCTS rollouts play games to completion
- Azul games average 107.6 moves in random play
- This is much longer than typical board games
- Each move involves expensive legal action computation

### 2. Expensive Legal Action Computation
- Called 16,288 times during 150 simulations (108 calls per simulation)
- Each call processes all possible factory/color/destination combinations
- Involves pattern line validation for each option

### 3. Frequent State Operations
- Heavy use of Action object hashing and equality checks
- Pattern line validation called 613,660 times
- Each validation involves tile placement logic

## Optimization Strategies (Ranked by Impact)

### 🥇 **Priority 1: Limit Rollout Length** (Expected: 2-3x speedup)
**Problem**: Rollouts average 107.6 moves, taking 5.8ms
**Solution**: Implement early termination with score estimation

```python
class LimitedRolloutEvaluator(mcts.Evaluator):
    def __init__(self, max_moves=40):  # Down from ~107 average
        self.max_moves = max_moves
    
    def _rollout(self, state):
        moves = 0
        while not state.is_terminal() and moves < self.max_moves:
            # ... rollout logic ...
            moves += 1
        
        if moves >= self.max_moves:
            return self._estimate_score(state)  # Use current score differential
        return actual_game_result
```

**Impact**: Reduces simulation time from 6.3ms to ~2-3ms

### 🥈 **Priority 2: Cache Legal Actions** (Expected: 1.5-2x speedup)
**Problem**: Legal actions computed 108 times per simulation
**Solution**: Cache legal actions until state changes

```python
class CachedGameState:
    def __init__(self):
        self._legal_actions_cache = None
        self._cache_valid = False
    
    def get_legal_actions(self):
        if not self._cache_valid:
            self._legal_actions_cache = self._compute_legal_actions()
            self._cache_valid = True
        return self._legal_actions_cache
    
    def apply_action(self, action):
        super().apply_action(action)
        self._cache_valid = False  # Invalidate cache
```

### 🥉 **Priority 3: Optimize Pattern Line Validation** (Expected: 1.2-1.5x speedup)
**Problem**: Pattern line validation called 613,660 times
**Solution**: Pre-compute valid placements or use faster lookup tables

### 🏅 **Priority 4: Action Object Optimization** (Expected: 1.1-1.2x speedup)
**Problem**: Heavy Action hashing and equality checking
**Solution**: Use integer action encoding or optimize hash function

## Implementation Plan

### Step 1: Quick Win - Rollout Limitation
1. Create `FastRolloutEvaluator` with 30-40 move limit
2. Use current score differential for estimation
3. Test with existing agent infrastructure

### Step 2: Legal Actions Caching
1. Add caching to `GameState.get_legal_actions()`
2. Invalidate cache on state changes
3. Measure performance improvement

### Step 3: Pattern Line Optimization
1. Profile pattern line validation in detail
2. Pre-compute valid tile placement options
3. Use lookup tables for common cases

### Step 4: Comprehensive Testing
1. Measure play quality vs baseline
2. Test in actual games
3. Validate performance improvements

## Expected Results

### Current Performance
- **Time per simulation**: 6.3ms
- **100 simulations**: 630ms
- **Simulations per second**: 158

### After Rollout Optimization (Priority 1)
- **Time per simulation**: 2-3ms
- **100 simulations**: 200-300ms
- **Simulations per second**: 330-500
- **Speedup**: 2-3x

### After All Optimizations
- **Time per simulation**: 1-2ms
- **100 simulations**: 100-200ms
- **Simulations per second**: 500-1000
- **Speedup**: 3-6x

## Quality Considerations

### Why Rollout Limitation Won't Hurt Quality:
1. **Azul scoring is continuous**: Points scored throughout the game
2. **30-40 moves covers most of game**: Usually sufficient for evaluation
3. **Score differential is meaningful**: Current score indicates trajectory
4. **MCTS tree search compensates**: Multiple shallow rollouts better than few deep ones

### Testing Quality:
1. Compare limited vs unlimited rollout agents
2. Measure win rates against baseline
3. Analyze move quality in real games

## Tools Created

### Profiling Scripts:
- `profile_mcts_simulation.py`: Basic simulation profiling
- `detailed_mcts_profiler.py`: Comprehensive component analysis
- `simple_optimized_mcts.py`: Rollout optimization testing

### Usage:
```bash
# Profile current performance
python detailed_mcts_profiler.py

# Test optimizations
python simple_optimized_mcts.py
```

## Conclusion

The 6.3ms simulation time is primarily caused by **extremely long random rollouts** (107.6 moves average). By limiting rollouts to 30-40 moves and estimating scores for incomplete games, we can achieve a **2-3x speedup** while maintaining good play quality.

The optimization is straightforward to implement and has minimal risk since:
1. Azul scoring happens throughout the game
2. Current score differential is a good proxy for final outcome
3. MCTS tree search will compensate for shorter rollouts

**Recommended immediate action**: Implement rollout length limitation as the first optimization step. 