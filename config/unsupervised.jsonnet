# Fine-tune a udify model on unlabeled data with typology supervision
local base = import 'base.libsonnet';
local loss_cfg = import 'typo_loss_specs.libsonnet';
local validation_metric = std.extVar('validation_metric');

base+{
  dataset_reader+: {
    for_min_risk: true,
  },
  validation_dataset_reader+: {
    for_min_risk: true,
  },

  model+: {
    loss_cfg: loss_cfg
  },
    
  trainer+: {
    validation_metric: validation_metric
  },
}