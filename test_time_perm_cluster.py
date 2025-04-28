import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, cpu_count
from ieeg.calc.fast import mean_diff

def iterate_axes(arr: np.ndarray, axes: tuple[int, ...]):
    """Iterate over all possible indices for a set of axes in a more controlled way.

    Parameters
    ----------
    arr : np.ndarray
        The array to iterate over
    axes : tuple[int]
        The axes to iterate over

    Yields
    ------
    tuple[slice]
        The indices for the current iteration
    """
    # Get the shape of the specified axes
    shapes = [arr.shape[axis] for axis in axes]
    
    # Generate all index combinations for the specified axes
    indices = np.ndindex(*shapes)
    
    # Create full slicing tuples
    for idx in indices:
        # Create a list of slices (all dimensions set to slice(None))
        slices = [slice(None)] * arr.ndim
        
        # Replace slices at specified axes with the corresponding index
        for i, axis in enumerate(axes):
            slices[axis] = idx[i]
        
        yield tuple(slices)

# Mock function for time_perm_cluster testing
def _proc(sig1, sig2, axis=0):
    # Simple mock function that returns a boolean mask and p-values
    # Make sure shape is correct (remove the axis dimension)
    out_shape = list(sig1.shape)
    if axis < len(out_shape):
        out_shape.pop(axis)
    result_mask = np.random.randint(0, 2, out_shape, dtype=bool)
    result_pvals = np.random.random(out_shape)
    return result_mask, result_pvals

def time_perm_cluster(sig1: np.ndarray, sig2: np.ndarray, p_thresh: float = 0.05,
                      p_cluster: float = None, n_perm: int = 100, 
                      tails: int = 1, axis: int = 0,
                      stat_func = mean_diff,
                      ignore_adjacency: tuple[int] | int = None,
                      n_jobs: int = 1, seed: int = None):
    """Modified time_perm_cluster function to test the new iterate_axes implementation.
    
    This function calculates significant clusters using permutation testing and
    cluster correction between two signals.
    
    Parameters
    ----------
    sig1 : np.ndarray
        First signal to compare
    sig2 : np.ndarray
        Second signal to compare
    p_thresh : float
        P-value threshold for significance
    p_cluster : float
        P-value threshold for cluster significance
    n_perm : int
        Number of permutations
    tails : int
        Number of tails (1 or 2)
    axis : int
        Axis along which to perform permutation
    stat_func : callable
        Statistical function to use
    ignore_adjacency : int or tuple of ints
        Axes to process independently
    n_jobs : int
        Number of parallel jobs
    seed : int
        Random seed
        
    Returns
    -------
    mask : np.ndarray
        Boolean mask of significant clusters
    pvals : np.ndarray
        P-values for each point
    """
    
    # Simple validation checks
    if p_cluster is None:
        p_cluster = p_thresh
        
    if isinstance(ignore_adjacency, int):
        ignore_adjacency = (ignore_adjacency,)
        
    if isinstance(ignore_adjacency, tuple):
        print(f"Processing with ignore_adjacency={ignore_adjacency}")
        # Calculate expected number of processes
        nprocs = np.prod([sig1.shape[i] for i in ignore_adjacency])
        print(f"Expected number of processes: {nprocs}")
        n_jobs = min(n_jobs, cpu_count())
    else:
        nprocs = 1
        
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # For parallel processing with ignored adjacency
    if nprocs > 1:
        # Create output shape - remove the axis dimension
        out_shape = list(sig1.shape)
        if axis < len(out_shape):
            out_shape.pop(axis)
        out1 = np.zeros(out_shape, dtype=int)
        out2 = np.zeros(out_shape, dtype=float)
        
        print(f"Output shape (excluding axis={axis}): {out_shape}")
        
        # Create iterators for the ignored dimensions
        slices = list(iterate_axes(sig1, ignore_adjacency))
        print(f"Number of slices generated: {len(slices)}")
        
        # Check that we have the expected number of slices
        expected_slices = np.prod([sig1.shape[i] for i in ignore_adjacency])
        assert len(slices) == expected_slices, f"Expected {expected_slices} slices, got {len(slices)}"
        
        # Each slice will be a tuple of indices/slices for all dimensions
        ins = [(sig1[sl], sig2[sl]) for sl in slices]
        
        # Adjust axis for dimension removal
        axis_adjusted = axis
        for i in ignore_adjacency:
            if i < axis:
                axis_adjusted -= 1
        
        # Process in parallel (or sequentially for testing)
        results = []
        for i, (s1, s2) in enumerate(ins):
            results.append(_proc(s1, s2, axis=axis_adjusted))
        
        # Map results back to output arrays
        for i, (mask, pvals) in enumerate(results):
            # Convert flat index i to multi-dimensional indices for ignored dimensions
            idx_tuple = np.unravel_index(i, [sig1.shape[ax] for ax in ignore_adjacency])
            
            # Build output index tuple for this result
            output_idx = []
            ignore_idx_pos = 0
            
            # Current dimension in the output shape (accounting for axis removal)
            out_dim = 0
            
            # Iterate through original dimensions
            for orig_dim in range(sig1.ndim):
                # Skip the axis dimension
                if orig_dim == axis:
                    continue
                
                # If this is an ignored dimension, use the corresponding index
                if orig_dim in ignore_adjacency:
                    output_idx.append(idx_tuple[ignore_idx_pos])
                    ignore_idx_pos += 1
                else:
                    # Otherwise, take all elements
                    output_idx.append(slice(None))
                
                out_dim += 1
            
            # Convert to tuple for indexing
            output_idx = tuple(output_idx)
            
            # Assign results to output arrays
            out1[output_idx] = mask
            out2[output_idx] = pvals
        
        return out1, out2
    else:
        # For non-parallel case
        return _proc(sig1, sig2, axis=axis)

# Test cases
def test_iterate_axes():
    print("\n=== Testing iterate_axes function ===")
    
    # Test with a 3D array
    arr = np.arange(24).reshape(2, 3, 4)
    print(f"Original array shape: {arr.shape}")
    
    # Test with single axis
    slices = list(iterate_axes(arr, (0,)))
    print(f"Number of slices with axes=(0,): {len(slices)}")
    assert len(slices) == arr.shape[0], f"Expected {arr.shape[0]} slices, got {len(slices)}"
    
    # Test with multiple axes
    slices = list(iterate_axes(arr, (0, 1)))
    print(f"Number of slices with axes=(0, 1): {len(slices)}")
    assert len(slices) == arr.shape[0] * arr.shape[1], f"Expected {arr.shape[0] * arr.shape[1]} slices, got {len(slices)}"
    
    # Verify correct indexing
    for i, sl in enumerate(slices):
        # Check the first few slices
        if i < 3:
            print(f"Slice {i}: {sl}")
            # Extract the indexed values
            value = arr[sl]
            print(f"  Extracted shape: {value.shape}")
    
    print("iterate_axes test completed successfully!")

def test_time_perm_cluster():
    print("\n=== Testing time_perm_cluster function ===")
    
    # Create test data
    sig1 = np.random.random((20, 4, 5, 100))  # trials x channels x frequencies x time
    sig2 = np.random.random((20, 4, 5, 100))
    
    print(f"Input shapes - sig1: {sig1.shape}, sig2: {sig2.shape}")
    
    # Test with single axis to ignore
    ignore_adjacency = 1  # Ignore channel dimension
    print(f"\nTesting with ignore_adjacency={ignore_adjacency}")
    
    mask, pvals = time_perm_cluster(
        sig1, sig2, 
        ignore_adjacency=ignore_adjacency, 
        n_jobs=1,
        axis=0,  # Specify axis=0 (trials dimension)
        seed=42
    )
    
    # Expected shape: (4, 5, 100) - channels, frequencies, time (no trials dimension)
    print(f"Output shapes - mask: {mask.shape}, pvals: {pvals.shape}")
    expected_shape = (4, 5, 100)
    assert mask.shape == expected_shape, f"Expected shape {expected_shape}, got {mask.shape}"
    
    # Test with multiple axes to ignore
    ignore_adjacency = (1, 2)  # Ignore channel and frequency dimensions
    print(f"\nTesting with ignore_adjacency={ignore_adjacency}")
    
    mask, pvals = time_perm_cluster(
        sig1, sig2, 
        ignore_adjacency=ignore_adjacency, 
        n_jobs=1,
        axis=0,  # Specify axis=0 (trials dimension)
        seed=42
    )
    
    # Expected shape: (4, 5, 100) - (channels, frequencies, time)
    print(f"Output shapes - mask: {mask.shape}, pvals: {pvals.shape}")
    expected_shape = (4, 5, 100)
    assert mask.shape == expected_shape, f"Expected shape {expected_shape}, got {mask.shape}"
    
    print("time_perm_cluster test completed successfully!")

# Run the tests
if __name__ == "__main__":
    test_iterate_axes()
    test_time_perm_cluster()