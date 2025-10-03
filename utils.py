import torch
import pandas as pd

def prepare_data(df, context_len, freq='D'):
    """Prepare data for inference."""
    # This is a placeholder for actual Groups (from a library like `statsforecast`)
    # For a single series, we can simulate this structure.
    class MockGroups:
        def __init__(self, data, indptr):
            self.data = data
            self.indptr = indptr

    # Ensure ds is datetime
    df['ds'] = pd.to_datetime(df['ds'])

    # Get the last `context_len` points
    y_context = df['y'].values[-context_len:]

    # Create a batch
    batch_size = 1
    contexts = torch.from_numpy(y_context).float().unsqueeze(0)
    
    # Simulate the Groups object that the original code might expect
    G_df = MockGroups(data=y_context, indptr=torch.tensor([0, len(y_context)]))

    return contexts, G_df