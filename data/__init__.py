"""Dataset registry."""

from data.acdc import ACDCDataset
from data.mnms2 import MnMs2Dataset

DATASETS = {
    'acdc': ACDCDataset,
    'mnms2': MnMs2Dataset,
}
