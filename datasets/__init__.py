from datasets.optical_flow_datasets.mpisintel import mpi_sintel_clean, mpi_sintel_final, mpi_sintel, MPISintelTestData
from datasets.geometric_matching_datasets.hpatches import HPatchesdataset
from datasets.geometric_matching_datasets.training_dataset import HomoAffTpsDataset
from datasets.semantic_matching_datasets.tss import TSSDataset
from datasets.optical_flow_datasets.KITTI_optical_flow import KITTI_noc, KITTI_occ, KITTI_only_occ
from datasets.semantic_matching_datasets.pfpascal import PFPascalDataset
from datasets.semantic_matching_datasets.pfwillow import PFWillowDataset
from datasets.semantic_matching_datasets.spair import SPairDataset
from datasets.semantic_matching_datasets.caltech_dataset import CaltechDataset
from datasets.geometric_matching_datasets.ETH3D_interval import ETHInterval

__all__ = ('KITTI_occ', 'KITTI_noc', 'KITTI_only_occ', 'mpi_sintel_clean', 'mpi_sintel',
           'mpi_sintel_final', 'SPairDataset', 'CaltechDataset',
           'MPISintelTestData', 'ETHInterval',
           'HPatchesdataset', 'HomoAffTpsDataset', 'TSSDataset', 'PFPascalDataset', 'PFWillowDataset')