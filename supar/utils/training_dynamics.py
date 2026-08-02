import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def log_training_dynamics(output_dir, epoch, train_ids, train_logits, train_golds):
    """Save training logits for a given epoch as JSONL records."""
    td_df = pd.DataFrame({
        "guid": train_ids,
        f"logits_epoch_{epoch}": train_logits,
        "gold": train_golds,
    })
    logging_dir = os.path.join(output_dir)
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    epoch_file_name = os.path.join(logging_dir, f"dynamics_epoch_{epoch}.jsonl")
    td_df.to_json(epoch_file_name, lines=True, orient="records")
    logger.info(f"\nTraining Dynamics logged to {epoch_file_name}\n")
