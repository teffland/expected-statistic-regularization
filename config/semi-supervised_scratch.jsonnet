# Fine-tune a udify model on supervised and unsupervised data, using typology loss on unsupervised
# Fine-tune a udify model on unlabeled data with typology supervision
local base = import 'base.libsonnet';
local loss_cfg = import 'typo_loss_specs.libsonnet';
local validation_metric = std.extVar('validation_metric');
local num_unsup_per_sup = std.parseJson(std.extVar('num_unsup_per_sup'));
local supervised_train_data_path = std.extVar('supervised_train_data_path');
local unsupervised_train_data_path = std.extVar('unsupervised_train_data_path');
local unsup_loss_weight = std.parseJson(std.extVar('unsup_loss_weight'));

base+{
  train_data_path: {
      supervised: supervised_train_data_path,
      unsupervised: unsupervised_train_data_path,
  },
  dataset_reader: {
      type:"interleaving",
      scheme: "all_at_once",
      readers: {
          supervised: base.dataset_reader+{
              for_min_risk: false,
          },
          unsupervised: base.dataset_reader+{
              for_min_risk: true,
          },
      },
  },
  data_loader+: {
    batch_sampler+: {
      type: "multi_dataset_bucket",
      sample_proportions: {
          supervised: 1,
          unsupervised: num_unsup_per_sup,
      },
    },
  },
  validation_dataset_reader+: {
    for_min_risk: true,
  },
  model+: {
    initializer: null, // the only difference
    loss_cfg: loss_cfg+{
        total_weight: unsup_loss_weight,
    },
  },
    
  trainer+: {
    validation_metric: validation_metric
  },
}