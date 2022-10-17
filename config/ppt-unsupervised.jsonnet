# Train model unsupervised with ppt loss
local base = import 'base.libsonnet';
local validation_metric = std.extVar('validation_metric');
local train_data_limit = std.parseJson(std.extVar('train_data_limit'));

base+{
  dataset_reader+: {
    for_min_risk: true,
    skip_short: true,
    limit: train_data_limit,
  },
  validation_dataset_reader+: {
    for_min_risk: true,
    skip_short: false,
  },

  model+: {
    type: "ppt_udify_model",
  },
    
  trainer+: {
    validation_metric: validation_metric,
    callbacks: base.trainer.callbacks+[{
      type:"ppt_masker"
    }],
  },
}