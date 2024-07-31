import os
import hydra
import torch
from omegaconf import OmegaConf
from typing import Union
import pkg_resources

from torch_geometric.nn import global_max_pool

from graphium.finetuning.fingerprinting import Fingerprinter
from graphium.config._loader import (
    load_accelerator,
    load_predictor,
    load_metrics,
    load_architecture,
    load_datamodule,
)

from tqdm import tqdm

from torch_geometric.data import Batch


class Minimol: 
    
    def __init__(self):
        # handle the paths
        state_dict_path = pkg_resources.resource_filename('minimol.ckpts.minimol_v1', 'state_dict.pth')
        config_path     = pkg_resources.resource_filename('minimol.ckpts.minimol_v1', 'config.yaml')
        base_shape_path = pkg_resources.resource_filename('minimol.ckpts.minimol_v1', 'base_shape.yaml')

        # Load the config
        cfg = self.load_config(os.path.basename(config_path))
        cfg = OmegaConf.to_container(cfg, resolve=True)
        # Set the accelerator
        cfg['accelerator']['type'] = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.cfg, accelerator_type = load_accelerator(cfg)
        # Load the datamodule
        self.cfg['architecture']['mup_base_path'] = base_shape_path
        self.datamodule = load_datamodule(self.cfg, accelerator_type)
        # Load the model
        model_class, model_kwargs = load_architecture(cfg, in_dims=self.datamodule.in_dims)
        metrics = load_metrics(self.cfg)
        predictor = load_predictor(
            config=self.cfg,
            model_class=model_class,
            model_kwargs=model_kwargs,
            metrics=metrics,
            task_levels=self.datamodule.get_task_levels(),
            accelerator_type=accelerator_type,
            featurization=self.datamodule.featurization,
            task_norms=self.datamodule.task_norms,
            replicas=1,
            gradient_acc=1,
            global_bs=self.datamodule.batch_size_training,
        )
        predictor.load_state_dict(torch.load(state_dict_path), strict=False)
        self.predictor = Fingerprinter(predictor, 'gnn:15')
        self.predictor.setup()

    def load_config(self, config_name):
        hydra.initialize('ckpts/minimol_v1/', version_base=None)
        cfg = hydra.compose(config_name=config_name)
        return cfg

    def __call__(self, smiles: Union[str,list]) -> torch.Tensor:
        smiles = [smiles] if not isinstance(smiles, list) else smiles

        input_features, _ = self.datamodule._featurize_molecules(smiles)
        input_features = self.to_fp32(input_features)
        
        batch_size = min(100, len(input_features))

        results = []
        for i in tqdm(range(0, len(input_features), batch_size)):
            batch = Batch.from_data_list(input_features[i:(i + batch_size)])
            batch = {"features": batch, "batch_indices": batch.batch}
            node_features = self.predictor.get_fingerprints_for_batch(batch)
            fingerprint_graph = global_max_pool(node_features, batch['batch_indices'])
            num_molecules = min(batch_size, fingerprint_graph.shape[0])
            results += [fingerprint_graph[i] for i in range(num_molecules)]

        return results
    
    def to_fp32(self, input_features: list) -> list:
        failures = 0
        for input_feature in tqdm(input_features, desc="Casting to FP32"):
            try:
                if not isinstance(input_feature, str):
                    for k, v in input_feature.items():
                        if isinstance(v, torch.Tensor):
                            if v.dtype == torch.half:
                                input_feature[k] = v.float()
                            elif v.dtype == torch.int32:
                                input_feature[k] = v.long()
                else:
                    failures += 1
            except Exception as e:
                print(f"{input_feature = }")
                raise e

        if failures != 0:
            print(f"{failures = }")
        return input_features
