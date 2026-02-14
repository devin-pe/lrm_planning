"""
Quick test script to verify the Towers of Hanoi validator and dataset
without running the full training loop.
"""

from planning import (
    TowersOfHanoiValidator,
    TowersOfHanoiSolver,
    TowersOfHanoiDataset,
    TowersOfHanoiState
)

def test_state():
    """Test state representation and moves."""
    print("=" * 60)
    print("Testing State Representation")
    print("=" * 60)
    
    state = TowersOfHanoiState(num_disks=3)
    print(f"Initial state: {state}")
    
    # Valid move
    assert state.is_valid_move('A', 'C'), "Should be valid"
    state.apply_move('A', 'C')
    print(f"After A->C: {state}")
    
    # Invalid move (larger on smaller)
    assert not state.is_valid_move('A', 'C'), "Should be invalid"
    print("✓ Invalid move correctly rejected")
    
    print()

def test_solver():
    """Test optimal solver."""
    print("=" * 60)
    print("Testing Optimal Solver")
    print("=" * 60)
    
    solver = TowersOfHanoiSolver()
    
    for n in [2, 3, 4]:
        moves = solver.solve(n)
        expected_moves = 2**n - 1
        print(f"{n} disks: {len(moves)} moves (expected {expected_moves})")
        assert len(moves) == expected_moves, f"Wrong number of moves for {n} disks"
        
        # Verify solution is valid
        state = TowersOfHanoiState(n)
        for _, from_peg, to_peg in moves:
            assert state.is_valid_move(from_peg, to_peg), f"Invalid move: {from_peg}->{to_peg}"
            state.apply_move(from_peg, to_peg)
        
        assert state.is_goal('C'), f"Solution doesn't reach goal for {n} disks"
        print(f"  ✓ Solution verified for {n} disks")
    
    print()

def test_reasoning_trace():
    """Test reasoning trace generation."""
    print("=" * 60)
    print("Testing Reasoning Trace Generation")
    print("=" * 60)
    
    solver = TowersOfHanoiSolver()
    trace = solver.generate_reasoning_trace(3)
    
    print("Generated trace (first 500 chars):")
    print(trace[:500] + "...\n")
    
    assert "<think>" in trace, "Missing <think> tag"
    assert "</think>" in trace, "Missing </think> tag"
    assert "Move disk" in trace or "move disk" in trace, "Missing move instructions"
    print("✓ Trace format looks good")
    print()

def test_validator():
    """Test trace validation."""
    print("=" * 60)
    print("Testing Validator")
    print("=" * 60)
    
    validator = TowersOfHanoiValidator()
    solver = TowersOfHanoiSolver()
    
    # Test with correct trace
    correct_trace = solver.generate_reasoning_trace(3)
    problem_state = {'num_disks': 3, 'goal_peg': 'C'}
    reward, violations = validator.validate_trace(correct_trace, problem_state)
    
    print(f"Correct trace:")
    print(f"  Reward: {reward:.4f}")
    print(f"  Violations: {violations}")
    assert reward > 0.9, "Correct trace should have high reward"
    assert violations == 0, "Correct trace should have no violations"
    print("  ✓ Correct trace validated successfully")
    
    # Test with incorrect trace
    incorrect_trace = """<think>
    Move disk 3 from A to C
    Move disk 2 from A to B
    Move disk 3 from C to B
    </think>"""
    
    reward, violations = validator.validate_trace(incorrect_trace, problem_state)
    print(f"\nIncorrect trace:")
    print(f"  Reward: {reward:.4f}")
    print(f"  Violations: {violations}")
    assert violations > 0, "Incorrect trace should have violations"
    print("  ✓ Invalid moves correctly detected")
    
    print()

def test_dataset():
    """Test dataset generation."""
    print("=" * 60)
    print("Testing Dataset Generation")
    print("=" * 60)
    
    dataset = TowersOfHanoiDataset(min_disks=2, max_disks=4)
    
    for i in range(3):
        problem = dataset.generate_problem(num_disks=i+2)
        
        print(f"\nProblem {i+1} ({problem['num_disks']} disks):")
        print(f"  Optimal moves: {len(problem['optimal_moves'])}")
        print(f"  Expected: {2**problem['num_disks'] - 1}")
        
        assert len(problem['optimal_moves']) == 2**problem['num_disks'] - 1
        assert 'problem_text' in problem
        assert 'expert_trace' in problem
        print("  ✓ Problem generated correctly")
    
    # Test batch generation
    batch = dataset.generate_batch(5)
    assert len(batch) == 5, "Batch should have 5 problems"
    print(f"\n✓ Batch generation works ({len(batch)} problems)")
    
    print()

def test_batch_validation():
    """Test batch validation."""
    print("=" * 60)
    print("Testing Batch Validation")
    print("=" * 60)
    
    import torch
    
    validator = TowersOfHanoiValidator()
    dataset = TowersOfHanoiDataset(min_disks=2, max_disks=3)
    
    # Generate batch
    batch = dataset.generate_batch(4)
    traces = [item['expert_trace'] for item in batch]
    states = [{'num_disks': item['num_disks'], 'goal_peg': 'C'} for item in batch]
    
    # Validate
    rewards, violations = validator.batch_validate(traces, states)
    
    print(f"Batch validation results:")
    print(f"  Rewards: {rewards}")
    print(f"  Violations: {violations}")
    print(f"  Mean reward: {rewards.mean():.4f}")
    
    assert isinstance(rewards, torch.Tensor), "Should return tensor"
    assert len(rewards) == 4, "Should have 4 rewards"
    assert torch.all(rewards > 0.9), "All expert traces should have high reward"
    print("  ✓ Batch validation works correctly")
    
    print()

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RUNNING TOWERS OF HANOI VALIDATION TESTS")
    print("=" * 60 + "\n")
    
    try:
        test_state()
        test_solver()
        test_reasoning_trace()
        test_validator()
        test_dataset()
        test_batch_validation()
        
        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nThe validator is ready to use with the training script.")
        print("Run 'python training.py' to start training.\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        raise

if __name__ == "__main__":
    main()
