"""
Performance benchmark tests for Azul game implementation.

This module tests the performance characteristics of game operations,
state representation, and scalability under various conditions.
"""

import gc
import os
import time
from collections import defaultdict

import numpy as np
import pytest

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from azul_rl.game import AzulStateRepresentation, GameState


class TestBasicPerformance:
    """Test basic performance characteristics."""

    def test_game_creation_performance(self):
        """Test game creation speed across different player counts."""
        player_counts = [2, 3, 4]
        creation_times = {}

        for num_players in player_counts:
            # Warm up
            for _ in range(5):
                GameState(num_players=num_players, seed=42)

            # Benchmark
            start_time = time.time()
            num_iterations = 100

            for i in range(num_iterations):
                _ = GameState(num_players=num_players, seed=42 + i)

            end_time = time.time()
            avg_time = (end_time - start_time) / num_iterations
            creation_times[num_players] = avg_time

            # Should be very fast (< 1ms per game)
            assert (
                avg_time < 0.001
            ), f"Game creation too slow for {num_players} players: {avg_time:.6f}s"

        print(f"✓ Game creation performance: {creation_times}")

    def test_state_representation_creation_performance(self):
        """Test state representation creation speed."""
        # Create games with different complexities
        test_scenarios = [
            ("initial", 0),
            ("early_game", 5),
            ("mid_game", 15),
            ("late_game", 30),
        ]

        creation_times = {}

        for scenario_name, num_moves in test_scenarios:
            # Create game state
            game = GameState(num_players=3, seed=42)
            for _ in range(num_moves):
                actions = game.get_legal_actions()
                if actions:
                    game.apply_action(actions[0])
                else:
                    break

            # Warm up
            for _ in range(10):
                AzulStateRepresentation(game)

            # Benchmark
            start_time = time.time()
            num_iterations = 100

            for _ in range(num_iterations):
                _ = AzulStateRepresentation(game)

            end_time = time.time()
            avg_time = (end_time - start_time) / num_iterations
            creation_times[scenario_name] = avg_time

            # Should be fast (< 5ms per representation)
            assert (
                avg_time < 0.005
            ), f"State representation creation too slow for {scenario_name}: {avg_time:.6f}s"

        print(f"✓ State representation creation performance: {creation_times}")

    def test_flat_vector_generation_performance(self):
        """Test flat vector generation speed."""
        game = GameState(num_players=4, seed=42)

        # Create complex game state
        for _ in range(20):
            actions = game.get_legal_actions()
            if actions:
                game.apply_action(actions[0])

        state_repr = AzulStateRepresentation(game)

        # Test both normalized and unnormalized
        vector_types = [
            ("normalized", True),
            ("unnormalized", False),
        ]

        generation_times = {}

        for vector_type, normalize in vector_types:
            # Warm up
            for _ in range(10):
                state_repr.get_flat_state_vector(normalize=normalize)

            # Benchmark
            start_time = time.time()
            num_iterations = 1000

            for _ in range(num_iterations):
                _ = state_repr.get_flat_state_vector(normalize=normalize)

            end_time = time.time()
            avg_time = (end_time - start_time) / num_iterations
            generation_times[vector_type] = avg_time

            # Should be very fast (< 1ms per vector)
            assert (
                avg_time < 0.001
            ), f"Vector generation too slow for {vector_type}: {avg_time:.6f}s"

        print(f"✓ Flat vector generation performance: {generation_times}")

    def test_legal_actions_performance(self):
        """Test legal action generation speed."""
        game = GameState(num_players=3, seed=42)

        action_times = []

        # Test across different game states
        for move_num in range(50):
            # Warm up
            for _ in range(5):
                game.get_legal_actions()

            # Benchmark
            start_time = time.time()
            num_iterations = 100

            for _ in range(num_iterations):
                _ = game.get_legal_actions()

            end_time = time.time()
            avg_time = (end_time - start_time) / num_iterations
            action_times.append(avg_time)

            # Should be fast (< 2ms per call)
            assert (
                avg_time < 0.002
            ), f"Legal actions too slow at move {move_num}: {avg_time:.6f}s"

            # Apply an action to change state
            actions = game.get_legal_actions()
            if actions:
                game.apply_action(actions[0])
            else:
                break

        avg_action_time = np.mean(action_times)
        print(
            f"✓ Legal actions performance: {avg_action_time:.6f}s average ({len(action_times)} samples)"
        )

    def test_action_application_performance(self):
        """Test action application speed."""
        game = GameState(num_players=2, seed=42)

        application_times = []

        for move_num in range(30):
            actions = game.get_legal_actions()
            if not actions:
                break

            action = actions[0]

            # Create copy for benchmarking
            _ = game.copy()  # game_copy

            # Warm up
            for _ in range(5):
                test_game = game.copy()
                test_game.apply_action(action)

            # Benchmark
            start_time = time.time()
            num_iterations = 50

            for _ in range(num_iterations):
                test_game = game.copy()
                test_game.apply_action(action)

            end_time = time.time()
            avg_time = (end_time - start_time) / num_iterations
            application_times.append(avg_time)

            # Should be reasonably fast (< 5ms per action)
            assert (
                avg_time < 0.005
            ), f"Action application too slow at move {move_num}: {avg_time:.6f}s"

            # Apply action to main game
            game.apply_action(action)

        avg_application_time = np.mean(application_times)
        print(
            f"✓ Action application performance: {avg_application_time:.6f}s average ({len(application_times)} samples)"
        )


class TestScalabilityPerformance:
    """Test performance scaling with different parameters."""

    def test_player_count_scaling(self):
        """Test how performance scales with number of players."""
        player_counts = [2, 3, 4]
        scaling_results = {}

        for num_players in player_counts:
            # Test state representation creation
            game = GameState(num_players=num_players, seed=42)

            # Make some moves
            for _ in range(10):
                actions = game.get_legal_actions()
                if actions:
                    game.apply_action(actions[0])

            # Benchmark state representation
            start_time = time.time()
            num_iterations = 100

            for _ in range(num_iterations):
                state_repr = AzulStateRepresentation(game)
                _ = state_repr.get_flat_state_vector(normalize=True)

            end_time = time.time()
            avg_time = (end_time - start_time) / num_iterations
            scaling_results[num_players] = avg_time

        # Check that scaling is reasonable (should be roughly linear or better)
        time_2p = scaling_results[2]
        time_4p = scaling_results[4]
        scaling_factor = time_4p / time_2p

        # 4 players should not be more than 3x slower than 2 players
        assert (
            scaling_factor < 3.0
        ), f"Poor scaling: 4p is {scaling_factor:.2f}x slower than 2p"

        print(
            f"✓ Player count scaling: {scaling_results}, factor: {scaling_factor:.2f}"
        )

    def test_game_length_scaling(self):
        """Test how performance scales with game length."""
        _ = GameState(num_players=3, seed=42)  # game

        length_results = {}
        move_counts = [0, 10, 20, 30, 40]

        for target_moves in move_counts:
            # Reset game
            test_game = GameState(num_players=3, seed=42)

            # Make specified number of moves
            for _ in range(target_moves):
                actions = test_game.get_legal_actions()
                if actions:
                    test_game.apply_action(actions[0])
                else:
                    break

            # Benchmark state operations
            start_time = time.time()
            num_iterations = 50

            for _ in range(num_iterations):
                state_repr = AzulStateRepresentation(test_game)
                _ = state_repr.get_flat_state_vector(normalize=True)
                _ = test_game.get_legal_actions()

            end_time = time.time()
            avg_time = (end_time - start_time) / num_iterations
            length_results[target_moves] = avg_time

        # Performance should not degrade significantly with game length
        initial_time = length_results[0]
        final_time = length_results[max(move_counts)]
        degradation_factor = final_time / initial_time

        # Should not be more than 2x slower for longer games
        assert (
            degradation_factor < 2.0
        ), f"Performance degrades too much: {degradation_factor:.2f}x"

        print(
            f"✓ Game length scaling: {length_results}, degradation: {degradation_factor:.2f}x"
        )

    def test_batch_processing_performance(self):
        """Test performance when processing multiple games."""
        batch_sizes = [1, 10, 50, 100]
        batch_results = {}

        for batch_size in batch_sizes:
            # Create batch of games
            games = []
            for i in range(batch_size):
                game = GameState(num_players=2, seed=42 + i)
                # Make some moves
                for _ in range(5):
                    actions = game.get_legal_actions()
                    if actions:
                        game.apply_action(actions[0])
                games.append(game)

            # Benchmark batch processing
            start_time = time.time()

            state_vectors = []
            for game in games:
                state_repr = AzulStateRepresentation(game)
                vector = state_repr.get_flat_state_vector(normalize=True)
                state_vectors.append(vector)

            end_time = time.time()
            total_time = end_time - start_time
            per_game_time = total_time / batch_size
            batch_results[batch_size] = per_game_time

        # Per-game time should not increase significantly with batch size
        single_time = batch_results[1]
        batch_time = batch_results[max(batch_sizes)]
        overhead_factor = batch_time / single_time

        # Should not have more than 50% overhead for large batches
        assert (
            overhead_factor < 1.5
        ), f"Batch processing overhead too high: {overhead_factor:.2f}x"

        print(
            f"✓ Batch processing performance: {batch_results}, overhead: {overhead_factor:.2f}x"
        )


class TestMemoryPerformance:
    """Test memory usage and efficiency."""

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if not HAS_PSUTIL:
            pytest.skip("psutil not available")
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
    def test_memory_usage_scaling(self):
        """Test memory usage with different numbers of games."""
        gc.collect()  # Clean up before test
        _ = self.get_memory_usage()  # initial_memory

        game_counts = [1, 10, 50, 100]
        memory_results = {}

        for num_games in game_counts:
            gc.collect()
            start_memory = self.get_memory_usage()

            # Create games and state representations
            games = []
            state_reprs = []

            for i in range(num_games):
                game = GameState(num_players=3, seed=42 + i)
                # Make some moves
                for _ in range(8):
                    actions = game.get_legal_actions()
                    if actions:
                        game.apply_action(actions[0])

                state_repr = AzulStateRepresentation(game)
                games.append(game)
                state_reprs.append(state_repr)

            end_memory = self.get_memory_usage()
            memory_used = end_memory - start_memory
            memory_per_game = memory_used / num_games
            memory_results[num_games] = memory_per_game

            # Clean up
            del games
            del state_reprs
            gc.collect()

        # Memory per game should be reasonable (< 1MB per game)
        avg_memory_per_game = np.mean(list(memory_results.values()))
        assert (
            avg_memory_per_game < 1.0
        ), f"Memory usage too high: {avg_memory_per_game:.3f} MB per game"

        print(
            f"✓ Memory usage scaling: {memory_results}, average: {avg_memory_per_game:.3f} MB/game"
        )

    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated operations."""
        gc.collect()
        initial_memory = self.get_memory_usage()

        # Perform many operations
        for iteration in range(100):
            game = GameState(num_players=2, seed=42 + iteration)

            # Make moves
            for _ in range(10):
                actions = game.get_legal_actions()
                if actions:
                    game.apply_action(actions[0])

            # Create state representation
            state_repr = AzulStateRepresentation(game)
            _ = state_repr.get_flat_state_vector(normalize=True)  # vector

            # Check memory every 20 iterations
            if iteration % 20 == 19:
                gc.collect()
                current_memory = self.get_memory_usage()
                memory_growth = current_memory - initial_memory

                # Memory growth should be minimal (< 10MB)
                assert (
                    memory_growth < 10.0
                ), f"Potential memory leak: {memory_growth:.3f} MB growth at iteration {iteration}"

        gc.collect()
        final_memory = self.get_memory_usage()
        total_growth = final_memory - initial_memory

        print(f"✓ Memory leak test: {total_growth:.3f} MB total growth")

    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
    def test_large_state_memory_efficiency(self):
        """Test memory efficiency with large state representations."""
        # Create game with maximum complexity
        game = GameState(num_players=4, seed=42)

        # Play until complex state
        for _ in range(30):
            actions = game.get_legal_actions()
            if actions:
                game.apply_action(actions[0])

        gc.collect()
        start_memory = self.get_memory_usage()

        # Create multiple large state representations
        state_reprs = []
        for _ in range(10):
            state_repr = AzulStateRepresentation(game)
            vector = state_repr.get_flat_state_vector(normalize=True)
            state_reprs.append((state_repr, vector))

        end_memory = self.get_memory_usage()
        memory_used = end_memory - start_memory
        memory_per_state = memory_used / 10

        # Each state representation should use reasonable memory (< 100KB)
        assert (
            memory_per_state < 0.1
        ), f"State representation too large: {memory_per_state:.3f} MB"

        # Clean up
        del state_reprs
        gc.collect()

        print(f"✓ Large state memory efficiency: {memory_per_state:.3f} MB per state")


class TestConcurrentPerformance:
    """Test performance under concurrent conditions."""

    def test_independent_game_performance(self):
        """Test performance when running multiple independent games."""
        num_games = 20
        games = []

        # Create independent games
        start_time = time.time()

        for i in range(num_games):
            game = GameState(num_players=2, seed=42 + i)

            # Play each game independently
            for _ in range(15):
                actions = game.get_legal_actions()
                if actions:
                    game.apply_action(actions[0])
                else:
                    break

            games.append(game)

        end_time = time.time()
        total_time = end_time - start_time
        time_per_game = total_time / num_games

        # Should be efficient (< 50ms per game)
        assert (
            time_per_game < 0.05
        ), f"Independent games too slow: {time_per_game:.6f}s per game"

        print(f"✓ Independent game performance: {time_per_game:.6f}s per game")

    def test_state_representation_independence(self):
        """Test that state representations don't interfere with each other."""
        # Create multiple games
        games = []
        for i in range(10):
            game = GameState(num_players=3, seed=42 + i)
            for _ in range(8):
                actions = game.get_legal_actions()
                if actions:
                    game.apply_action(actions[0])
            games.append(game)

        # Create state representations simultaneously
        start_time = time.time()

        state_reprs = []
        vectors = []

        for game in games:
            state_repr = AzulStateRepresentation(game)
            vector = state_repr.get_flat_state_vector(normalize=True)
            state_reprs.append(state_repr)
            vectors.append(vector)

        end_time = time.time()
        total_time = end_time - start_time
        time_per_repr = total_time / len(games)

        # Should be fast (< 10ms per representation)
        assert (
            time_per_repr < 0.01
        ), f"Concurrent state representations too slow: {time_per_repr:.6f}s"

        # Verify independence - each vector should be different
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                assert not np.array_equal(
                    vectors[i], vectors[j]
                ), f"State vectors {i} and {j} are identical (should be independent)"

        print(
            f"✓ State representation independence: {time_per_repr:.6f}s per representation"
        )


class TestPerformanceRegression:
    """Test for performance regressions."""

    def test_baseline_performance_benchmarks(self):
        """Establish baseline performance benchmarks."""
        benchmarks = {}

        # Game creation benchmark
        start_time = time.time()
        for i in range(100):
            game = GameState(num_players=3, seed=42 + i)
        creation_time = (time.time() - start_time) / 100
        benchmarks["game_creation"] = creation_time

        # State representation benchmark
        game = GameState(num_players=3, seed=42)
        for _ in range(10):
            actions = game.get_legal_actions()
            if actions:
                game.apply_action(actions[0])

        start_time = time.time()
        for _ in range(100):
            state_repr = AzulStateRepresentation(game)
        repr_time = (time.time() - start_time) / 100
        benchmarks["state_representation"] = repr_time

        # Vector generation benchmark
        state_repr = AzulStateRepresentation(game)
        start_time = time.time()
        for _ in range(1000):
            _ = state_repr.get_flat_state_vector(normalize=True)  # vector
        vector_time = (time.time() - start_time) / 1000
        benchmarks["vector_generation"] = vector_time

        # Legal actions benchmark
        start_time = time.time()
        for _ in range(100):
            actions = game.get_legal_actions()
        actions_time = (time.time() - start_time) / 100
        benchmarks["legal_actions"] = actions_time

        # Verify all benchmarks are within reasonable limits
        limits = {
            "game_creation": 0.001,  # 1ms
            "state_representation": 0.005,  # 5ms
            "vector_generation": 0.001,  # 1ms
            "legal_actions": 0.002,  # 2ms
        }

        for operation, time_taken in benchmarks.items():
            limit = limits[operation]
            assert (
                time_taken < limit
            ), f"{operation} too slow: {time_taken:.6f}s > {limit}s"

        print(f"✓ Baseline benchmarks: {benchmarks}")
        # Store benchmarks for potential use by other tests
        self._last_benchmarks = benchmarks

    def test_performance_consistency(self):
        """Test that performance is consistent across multiple runs."""
        num_runs = 10
        operation_times = defaultdict(list)

        for run in range(num_runs):
            # Game creation
            start_time = time.time()
            game = GameState(num_players=3, seed=42 + run)
            operation_times["game_creation"].append(time.time() - start_time)

            # Make some moves
            for _ in range(5):
                actions = game.get_legal_actions()
                if actions:
                    game.apply_action(actions[0])

            # State representation
            start_time = time.time()
            state_repr = AzulStateRepresentation(game)
            operation_times["state_representation"].append(time.time() - start_time)

            # Vector generation
            start_time = time.time()
            _ = state_repr.get_flat_state_vector(normalize=True)  # vector
            operation_times["vector_generation"].append(time.time() - start_time)

        # Check consistency (coefficient of variation < 70% for micro-operations)
        for operation, times in operation_times.items():
            mean_time = np.mean(times)
            std_time = np.std(times)
            cv = std_time / mean_time if mean_time > 0 else 0

            # Allow higher variance for very fast operations like vector generation
            max_cv = 0.7 if operation == "vector_generation" else 0.5
            assert cv < max_cv, f"{operation} performance too inconsistent: CV={cv:.3f}"

        print(f"✓ Performance consistency verified across {num_runs} runs")


if __name__ == "__main__":
    print("=== Azul Performance Benchmark Tests ===")

    # Run performance tests
    basic_tests = TestBasicPerformance()
    print("\n1. Basic Performance Tests...")
    basic_tests.test_game_creation_performance()
    basic_tests.test_state_representation_creation_performance()
    basic_tests.test_flat_vector_generation_performance()
    basic_tests.test_legal_actions_performance()
    basic_tests.test_action_application_performance()

    scalability_tests = TestScalabilityPerformance()
    print("\n2. Scalability Tests...")
    scalability_tests.test_player_count_scaling()
    scalability_tests.test_game_length_scaling()
    scalability_tests.test_batch_processing_performance()

    if HAS_PSUTIL:
        memory_tests = TestMemoryPerformance()
        print("\n3. Memory Performance Tests...")
        memory_tests.test_memory_usage_scaling()
        memory_tests.test_memory_leak_detection()
        memory_tests.test_large_state_memory_efficiency()

    concurrent_tests = TestConcurrentPerformance()
    print("\n4. Concurrent Performance Tests...")
    concurrent_tests.test_independent_game_performance()
    concurrent_tests.test_state_representation_independence()

    regression_tests = TestPerformanceRegression()
    print("\n5. Performance Regression Tests...")
    regression_tests.test_baseline_performance_benchmarks()
    regression_tests.test_performance_consistency()

    print("\nAll performance tests completed successfully!")
    if hasattr(regression_tests, "_last_benchmarks"):
        print(f"Final benchmarks: {regression_tests._last_benchmarks}")
