# Azul Self-Play Performance Profiling Analysis

## Executive Summary

Your Azul self-play is taking **400+ seconds** because of a critical performance bottleneck: **game state copying**. The profiling reveals that 83% of execution time is spent in Python's `deepcopy()` function, making it 40-120x slower than it should be.

## Key Findings

### 🚨 Critical Bottleneck: Game State Copying
- **Time spent**: 513s out of 618s total (83% of execution time)
- **Operations**: 45,705 copy operations performed
- **Average cost**: ~11ms per copy operation
- **Root cause**: Python's `deepcopy()` is extremely expensive for complex game states
- **Location**: `game/game_state.py:280` - `copy()` method

### ✅ Neural Network Performance (Not the bottleneck)
- **Time spent**: 45.8s total (7.4% of execution time)
- **Operations**: 2,200 NN evaluations
- **Average cost**: ~20.8ms per evaluation
- **Breakdown**:
  - State conversion: 7.9s (17% of NN time)
  - Actual inference: 37.5s (82% of NN time)
- **Status**: Performance is reasonable, not the main issue

### ✅ Game Logic Performance (Acceptable)
- **get_legal_actions()**: 48.3s total, 50,905 calls (~0.95ms per call)
- **apply_action()**: Included in copy operations timing
- **Status**: Acceptable performance for game logic

### 🔍 MCTS Performance (Affected by copying)
- **MCTS search**: 612.6s total
- **Most time in**: expand_and_evaluate (611.5s)
- **Root cause**: MCTS creates many game state copies during tree search
- **Status**: MCTS algorithm itself is fine, but copying kills performance

## Detailed Profiling Results

### Timing Breakdown (10 simulations, 1 game)
```
Operation                      Total(s)   Avg(ms)    Count    Calls/s   
--------------------------------------------------------------------------------
total_self_play                617.930    617930.05  1        0.0       
self_play.full_game            617.930    617929.87  1        0.0       
mcts.full_search               612.597    3062.99    200      0.3       
mcts.expand_and_evaluate       611.491    277.95     2200     3.6       
mcts.single_simulation         555.117    277.56     2000     3.6       
nn.forward_pass                45.769     20.80      2200     48.1      
nn.inference                   37.483     17.04      2200     58.7      
nn.state_conversion            7.869      3.58       2200     279.6     
mcts.action_selection          0.195      0.09       2248     11547.4   
mcts.backpropagate             0.084      0.04       2000     23671.8   
```

### cProfile Hotspots
```
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
45705   8.692   0.000    513.008  0.011   game/game_state.py:280(copy)
33036072/45705  188.574  0.000  504.316  0.011  copy.py:128(deepcopy)
5623150/45705   72.957   0.000  502.602  0.011  copy.py:258(_reconstruct)
```

## Optimization Recommendations (Priority Order)

### 🥇 PRIORITY 1: Fix Game State Copying (CRITICAL)
**Expected improvement**: 5-10x speedup (from 618s to 60-120s)

**Current problem**: Using Python's `deepcopy()` - extremely slow
**Solutions**:
1. **Implement custom copy() method** for GameState
2. **Use copy-on-write** data structures
3. **Consider immutable** game state representation
4. **Cache intermediate states** in MCTS
5. **Use state deltas** instead of full copies

**Implementation**:
```python
def copy(self):
    # Replace deepcopy with manual field copying
    new_state = GameState.__new__(GameState)
    new_state.num_players = self.num_players
    new_state.current_player = self.current_player
    # ... copy other simple fields
    
    # Shallow copy where safe, deep copy only when necessary
    new_state.players = [player.copy() for player in self.players]
    new_state.factory_area = self.factory_area.copy()
    # ... etc
```

### 🥈 PRIORITY 2: Optimize MCTS State Management
**Expected improvement**: 2-3x additional speedup

**Solutions**:
1. **State pooling/reuse** - reuse GameState objects
2. **Transposition tables** - cache identical states
3. **Incremental state updates** - apply/undo actions instead of copying
4. **State hashing** for duplicate detection

### 🥉 PRIORITY 3: Neural Network Optimizations
**Expected improvement**: 2-4x speedup for NN component

**Current**: 20.8ms per evaluation (acceptable but can improve)
**Solutions**:
1. **Use MPS acceleration** (Apple Silicon GPU) - you have this available!
2. **Batch multiple evaluations** together
3. **Cache NN evaluations** for identical states
4. **Model quantization** for faster inference

### 🏅 PRIORITY 4: Game Logic Optimizations
**Expected improvement**: 20-50% speedup for game logic

**Solutions**:
1. **Cache legal actions** when game state hasn't changed
2. **Incremental action generation**
3. **Pre-compute common action patterns**

## Performance Projections

### Current Performance (1 game, 10 simulations)
- **Total time**: 618 seconds (~10 minutes)
- **Time per move**: ~3.1 seconds
- **Time per MCTS search**: ~3.1 seconds

### After Priority 1 (fix copying)
- **Estimated total time**: 60-120 seconds
- **Time per move**: 0.3-0.6 seconds
- **Speedup**: 5-10x improvement

### After Priorities 1+2 (copying + MCTS)
- **Estimated total time**: 20-40 seconds
- **Time per move**: 0.1-0.2 seconds
- **Speedup**: 15-30x improvement

### With all optimizations
- **Estimated total time**: 5-15 seconds
- **Time per move**: 0.025-0.075 seconds
- **Speedup**: 40-120x improvement

## Implementation Steps

### Step 1: Fix GameState.copy() (CRITICAL)
1. **Analyze current copy method** in `game/game_state.py:280`
2. **Implement custom copying** for each field
3. **Use shallow copies** where safe (immutable data)
4. **Profile the improvement** - should see 5-10x speedup

### Step 2: Optimize MCTS State Handling
1. **Implement state pooling** in MCTS
2. **Add transposition table** for duplicate states
3. **Consider incremental updates** (apply/undo actions)

### Step 3: Add MPS Acceleration
1. **Update neural network** to use MPS device
2. **Implement batch evaluation** for multiple states
3. **Add MPS profiling** to monitor GPU utilization

### Step 4: Profile and Iterate
1. **Re-run profiling** after each optimization
2. **Measure actual vs expected** improvements
3. **Identify new bottlenecks** as they emerge

## Immediate Action Items

1. **Fix GameState.copy() method** (`game/game_state.py:280`) - This is the biggest win
2. **Profile with MPS acceleration** for neural network
3. **Implement MCTS state pooling**
4. **Add batch NN evaluation**
5. **Re-run profiling** to measure improvements

## Tools and Scripts Created

### Profiling Tools
- `profiling/performance_profiler.py` - Comprehensive profiler
- `scripts/profile_self_play.py` - Self-play profiling script

### Analysis Tools
- `scripts/analyze_profiling_results.py` - Detailed analysis
- `scripts/test_gpu_profiling.py` - GPU/MPS testing

### Usage
```bash
# Install profiling dependencies
pip install -r requirements-dev.txt

# Run profiling (start with small tests)
python scripts/profile_self_play.py --games 1 --simulations 10 --nn-config small

# Analyze results
python scripts/analyze_profiling_results.py

# Test GPU/MPS performance
python scripts/test_gpu_profiling.py
```

## Conclusion

The 400-second self-play time is primarily due to inefficient game state copying (83% of time). By implementing a custom copy method, you can achieve a 5-10x speedup immediately. Combined with MCTS optimizations and MPS acceleration, you can potentially achieve 40-120x overall improvement, bringing self-play time down from 400+ seconds to 5-15 seconds.

**Focus on game state copying first** - it's the biggest bottleneck and will provide the most dramatic improvement. 