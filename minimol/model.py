import os
import hydra
import torch
from omegaconf import OmegaConf
from typing import Union
import pkg_resources
from contextlib import redirect_stdout, redirect_stderr

from torch_geometric.nn import global_max_pool

from graphium.trainer.predictor import PredictorModule
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
    
    def __init__(self, batch_size: int = 100, base_path: str = None):
        self.batch_size = batch_size
        # handle the paths
        if base_path is None:
            state_dict_path = pkg_resources.resource_filename('minimol.ckpts.minimol_v1', 'state_dict.pth')
            config_path     = pkg_resources.resource_filename('minimol.ckpts.minimol_v1', 'config.yaml')
            base_shape_path = pkg_resources.resource_filename('minimol.ckpts.minimol_v1', 'base_shape.yaml')
        else:
            ckpt_path       = os.path.join(base_path, 'model.ckpt')
            config_path     = os.path.join(base_path, 'config.yaml')
            base_shape_path = os.path.join(base_path, 'base_shape.yaml')
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

        if base_path is None:
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
        else:
            predictor = PredictorModule.load_pretrained_model(name_or_path=ckpt_path, device=accelerator_type)

        self.predictor = Fingerprinter(predictor, 'graph_output_nn-graph:0')
        self.predictor.setup()


    def load_config(self, config_name):
        hydra.initialize('ckpts/minimol_v1/', version_base=None)
        cfg = hydra.compose(config_name=config_name)
        return cfg

    def __call__(self, smiles: Union[str,list]) -> torch.Tensor:
        smiles = [smiles] if not isinstance(smiles, list) else smiles
        
        batch_size = min(self.batch_size, len(smiles))

        results = []
        for i in tqdm(range(0, len(smiles), batch_size)):
            with open(os.devnull, 'w') as fnull, redirect_stdout(fnull), redirect_stderr(fnull): # suppress output
                input_features, _ = self.datamodule._featurize_molecules(smiles[i:(i + batch_size)])
                input_features = self.to_fp32(input_features)

            batch = Batch.from_data_list(input_features)
            batch = {"features": batch, "batch_indices": batch.batch}
            fingerprint_graph = self.predictor.get_fingerprints_for_batch(batch)
            # node_features = self.predictor.get_fingerprints_for_batch(batch)
            # fingerprint_graph = global_max_pool(node_features, batch['batch_indices'])
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
