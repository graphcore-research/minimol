import hydra
import torch
from omegaconf import OmegaConf

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
    
    def __init__(self, config_path="../ckpts/minimol_v1", config_name="config"):
        # Load the config
        cfg = self.load_config(config_path, config_name)
        cfg = OmegaConf.to_container(cfg, resolve=True)
        # Set the accelerator
        cfg['accelerator']['type'] = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.cfg, accelerator_type = load_accelerator(cfg)
        # Load the datamodule
        self.datamodule = load_datamodule(self.cfg, accelerator_type)
        # Load the model
        model_class, model_kwargs = load_architecture(cfg, in_dims=self.datamodule.in_dims)
        metrics = load_metrics(self.cfg)
        self.predictor = load_predictor(
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
        self.predictor.load_state_dict(torch.load(f"{config_path}/state_dict.pth"))

    def load_config(self, config_path, config_name):
        hydra.initialize(config_path=config_path, version_base=None)
        cfg = hydra.compose(config_name=config_name)
        return cfg

    def __call__(self, smiles: list) -> torch.Tensor:   
        input_features, _ = self.datamodule._featurize_molecules(smiles)
        input_features = self.to_fp32(input_features)

        batch_size = min(100, len(input_features))

        results = []
        for i in tqdm(range(0, len(input_features), batch_size)):
            batch = Batch.from_data_list(input_features[i:(i + batch_size)])
            model_fp32 = self.predictor.model.float()
            _, extras = model_fp32.forward(batch, extra_return_names=["pre_task_heads"])
            fingerprint_graph = extras['pre_task_heads']['graph_feat']
            num_molecules = min(batch_size, fingerprint_graph.shape[0])
            results += [fingerprint_graph[i].detach().numpy() for i in range(num_molecules)]
        
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
