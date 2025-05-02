import os

import numpy as np
import pandas as pd

def load_data(data_path):
    """Load data from CSV file."""
    df = pd.read_csv(data_path)
    energies = df['Energy_eV'].to_numpy()
    data = df['Absorption'].to_numpy()
    return energies, data

def uniform_sparse_sampling(full_grid, sampling_fraction=0.1):
    """
    Create a uniform sparse sampling scheme that selects points at regular intervals,
    always including the first and last points of the full grid.
    
    Parameters:
    -----------
    full_grid : numpy.ndarray
        Full energy grid for XANES
    sampling_fraction : float
        Fraction of points to sample from the full grid
    
    Returns:
    --------
    numpy.ndarray
        Uniformly sampled sparse grid
    """
    # Calculate the number of points to sample (minus 2 for the endpoints we'll add)
    n_interior_samples = max(0, int(len(full_grid) * sampling_fraction) - 2)
    
    if n_interior_samples <= 0:
        # If sampling fraction is very low, just return endpoints
        return np.array([full_grid[0], full_grid[-1]])
    
    # For uniform sampling of interior points
    # Select n_interior_samples points from the interior of the grid (excluding first and last)
    interior_grid = full_grid[1:-1]
    
    # Calculate indices for uniform sampling of interior points
    if len(interior_grid) <= n_interior_samples:
        # If we need most interior points, just take them all
        indices = np.arange(len(interior_grid))
    else:
        # Calculate the stride needed to get approximately n_interior_samples
        stride = len(interior_grid) / n_interior_samples
        # Generate indices at approximately uniform intervals
        indices = np.round(np.arange(0, len(interior_grid), stride)).astype(int)
        # Ensure we don't exceed the array bounds
        indices = indices[indices < len(interior_grid)]
    
    # Select interior points
    selected_interior = interior_grid[indices]
    
    # Combine with endpoints
    uniform_grid = np.concatenate([[full_grid[0]], selected_interior, [full_grid[-1]]])
    
    return uniform_grid

def get_sparse_data(sparse_grid, full_grid, full_data):
    """
    Get absorption values corresponding to sparse energy grid points.
    
    Parameters:
    -----------
    sparse_grid : numpy.ndarray
        Sparse energy grid points
    full_grid : numpy.ndarray
        Full energy grid points
    full_data : numpy.ndarray
        Absorption values corresponding to full energy grid
        
    Returns:
    --------
    numpy.ndarray
        Absorption values corresponding to sparse energy grid points
    """
    # Find indices of sparse grid points in full grid
    sparse_indices = np.searchsorted(full_grid, sparse_grid)
    
    # Get corresponding absorption values
    sparse_data = full_data[sparse_indices]
    
    return sparse_data

def create_submission(sparse_grid, sparse_values, filename='submission.csv'):
    sparse_data_dict = {
        'Energy_eV': sparse_grid,
        'Absorption': sparse_values
    }
    submission_df = pd.DataFrame(sparse_data_dict)
    submission_df.to_csv(filename, index=False)
    print(f"Submission file '{filename}' has been created.")

def main():
    # Load data
    energies, data = load_data('./data/xanes_spectrum.csv')
    
    # XANES sampling
    sparse_energies = uniform_sparse_sampling(energies, sampling_fraction=0.1)
    sparse_data = get_sparse_data(sparse_energies, energies, data)

    create_submission(sparse_energies, sparse_data)

if __name__ == "__main__":
    main()