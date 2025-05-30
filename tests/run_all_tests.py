#!/usr/bin/env python3
"""
Comprehensive test runner for all Azul game unit tests.
"""

import os
import subprocess
import sys
import time

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_test_file(test_file):
    """Run a single test file and return results."""
    print(f"\n{'='*60}")
    print(f"Running {test_file}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        end_time = time.time()
        duration = end_time - start_time

        print(f"Duration: {duration:.2f} seconds")
        print(f"Return code: {result.returncode}")

        if result.stdout:
            print("\nSTDOUT:")
            print(result.stdout)

        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)

        return {
            "file": test_file,
            "success": result.returncode == 0,
            "duration": duration,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except Exception as e:
        print(f"Error running {test_file}: {e}")
        return {"file": test_file, "success": False, "duration": 0, "error": str(e)}


def main():
    """Run all test files and provide summary."""
    print("Azul Game - Comprehensive Unit Test Suite")
    print("=" * 60)

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # List of test files to run (relative to the script directory)
    test_files = [
        os.path.join(script_dir, "test_scoring.py"),
        os.path.join(script_dir, "test_legal_moves.py"),
        os.path.join(script_dir, "test_tile_placement.py"),
        os.path.join(script_dir, "test_game_flow.py"),
        os.path.join(script_dir, "test_edge_cases.py"),
        os.path.join(script_dir, "test_basic_functionality.py"),
        os.path.join(script_dir, "test_state_representation.py"),
    ]

    results = []
    total_start_time = time.time()

    # Run each test file
    for test_file in test_files:
        if os.path.exists(test_file):
            result = run_test_file(test_file)
            results.append(result)
        else:
            print(f"\nWarning: Test file {test_file} not found, skipping...")
            results.append(
                {
                    "file": test_file,
                    "success": False,
                    "duration": 0,
                    "error": "File not found",
                }
            )

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    successful_tests = [r for r in results if r["success"]]
    failed_tests = [r for r in results if not r["success"]]

    print(f"Total tests run: {len(results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Total duration: {total_duration:.2f} seconds")

    if successful_tests:
        print("\n✅ SUCCESSFUL TESTS:")
        for result in successful_tests:
            print(f"  - {os.path.basename(result['file'])} ({result['duration']:.2f}s)")

    if failed_tests:
        print("\n❌ FAILED TESTS:")
        for result in failed_tests:
            print(f"  - {os.path.basename(result['file'])}")
            if "error" in result:
                print(f"    Error: {result['error']}")

    # Test coverage summary
    print(f"\n{'='*60}")
    print("TEST COVERAGE SUMMARY")
    print(f"{'='*60}")

    coverage_areas = {
        "Scoring Mechanics": "test_scoring.py",
        "Legal Move Generation": "test_legal_moves.py",
        "Tile Placement Rules": "test_tile_placement.py",
        "Game Flow & Transitions": "test_game_flow.py",
        "Edge Cases & Boundaries": "test_edge_cases.py",
        "Basic Functionality": "test_basic_functionality.py",
        "State Representation": "test_state_representation.py",
    }

    for area, test_file in coverage_areas.items():
        status = (
            "✅"
            if any(r["success"] and test_file in r["file"] for r in results)
            else "❌"
        )
        print(f"{status} {area}")

    # Detailed test categories
    print(f"\n{'='*60}")
    print("DETAILED TEST CATEGORIES")
    print(f"{'='*60}")

    test_categories = {
        "Scoring Tests": [
            "Wall scoring (individual placements, connections)",
            "Pattern line completion scoring",
            "Floor line penalties",
            "Final scoring bonuses (rows, columns, colors)",
            "Score boundary conditions (non-negative)",
        ],
        "Legal Move Tests": [
            "Basic legal move generation",
            "Pattern line constraints (capacity, color, wall)",
            "Factory and center constraints",
            "Complex mid-game scenarios",
            "Edge cases and invalid moves",
        ],
        "Tile Placement Tests": [
            "Pattern line placement rules",
            "Wall placement validation",
            "Overflow handling to floor line",
            "Color consistency enforcement",
            "First player marker handling",
        ],
        "Game Flow Tests": [
            "Round transitions and scoring",
            "Game termination conditions",
            "Turn management and progression",
            "First player marker transfer",
            "Complete game simulation",
        ],
        "Edge Case Tests": [
            "Boundary conditions (min/max players, indices)",
            "Empty state handling",
            "Invalid input handling",
            "Resource exhaustion scenarios",
            "Data integrity and tile conservation",
        ],
    }

    for category, tests in test_categories.items():
        print(f"\n{category}:")
        for test in tests:
            print(f"  • {test}")

    # Return exit code
    return 0 if len(failed_tests) == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
