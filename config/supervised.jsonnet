# Fine-tune a udify model to supervised data
local base = import 'base.libsonnet';
local validation_metric = std.extVar('validation_metric');

base+{
  dataset_reader+: {
    for_min_risk: false,
  },
  validation_dataset_reader+: {
    for_min_risk: false,
  },
  trainer+: {
    validation_metric: validation_metric
  },
}