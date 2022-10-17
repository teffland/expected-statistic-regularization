# Define the core model elements core to each of the different ways to transfer

# Config
local train_data_path = std.extVar('train_data_path');
local validation_data_path = std.extVar('validation_data_path');
local vocab_path = std.extVar("vocab_path");
local weights_file_path = std.extVar('weights_file_path');

local wandb_name = std.extVar('wandb_name');
local wandb_note = std.extVar('wandb_note');
local wandb_tag = std.extVar('wandb_tag');

local random_seed = std.parseJson(std.extVar('random_seed'));
local train_data_limit = std.parseJson(std.extVar('train_data_limit'));
local batch_size = std.parseJson(std.extVar('batch_size'));
local num_batches_per_epoch = std.parseJson(std.extVar('num_batches_per_epoch'));
local num_epochs = std.parseJson(std.extVar('num_epochs'));
local scheduler_num_epochs = std.parseJson(std.extVar('scheduler_num_epochs'));
local patience = std.parseJson(std.extVar('patience'));
local learning_rate = std.parseJson(std.extVar('learning_rate'));
local weight_decay = std.parseJson(std.extVar('weight_decay'));


{
  random_seed: random_seed,
  numpy_seed: random_seed,
  pytorch_seed: random_seed,

  train_data_path: train_data_path,
  validation_data_path: validation_data_path,
  evaluate_on_test: false,
  
  vocabulary: {
    type: "from_files",
    directory: vocab_path,
  },

  dataset_reader: {
    type: "udify_universal_dependencies",
    lazy: false,
    no_labels: false,
    limit: train_data_limit,
    token_indexers: {
      bert: {
        type: "udify-bert-pretrained",
        pretrained_model: "/home/ubuntu/syntax//config/bert-base-multilingual-cased/vocab.txt",
        do_lowercase: false,
        use_starting_offsets: true
      }
    }
  },

  validation_dataset_reader: {
    type: "udify_universal_dependencies",
    lazy: false,
    no_labels: false,
    token_indexers: {
      bert: {
        type: "udify-bert-pretrained",
        pretrained_model: "/home/ubuntu/syntax//config/bert-base-multilingual-cased/vocab.txt",
        do_lowercase: false,
        use_starting_offsets: true
      }
    }
  },

  data_loader: {
    num_workers: 0,
    batches_per_epoch: num_batches_per_epoch,
    batch_sampler: {
      type: "bucket",
      batch_size: batch_size,
    },
  },
  validation_data_loader: {
    num_workers: 0,
    batch_sampler: {
      type: "bucket",
      batch_size: batch_size,
    },
  },
  
  
  model: {
    type: "expected_syntax_udify_model",

    initializer: {
      // Inherit all weights except the dep label fine_to_coarse reduction matrix
      regexes: [
        [".*", 
          {
            type: "pretrained",
            weights_file_path: weights_file_path, 
          }]
      ],
      prevent_regexes: ["decoders\\.deps\\.dep_label_fine_to_coarse_reduce_matrix"],

    },
    tasks: [
        "upos",
        // "feats",
        // "lemmas",
        "deps"
    ],
    
    word_dropout: 0.2,
    mix_embedding: 12,
    layer_dropout: 0.1,
    local std_dropout = 0.5,

    text_field_embedder: {
      type: "udify_embedder",
      dropout: std_dropout,
      allow_unmatched_keys: true,
      embedder_to_indexer_map: {
        bert: [
          "bert",
          "bert-offsets"
        ]
      },
      token_embedders: {
        bert: {
          type: "udify-bert-pretrained",
          pretrained_model: "bert-base-multilingual-cased",
          requires_grad: true,
          dropout: 0.15,
          layer_dropout: 0.1,
          combine_layers: "all"
        }
      }
    },

    local pass_through_layer = {
      type: "pass_through",
      input_dim: 768
    },

    encoder: pass_through_layer,

    decoders: {
      upos: {
        type: "udify_tag_decoder",
        task: "upos",
        encoder: pass_through_layer,
        dropout: std_dropout,
      },
      // feats: {
      //   type: "udify_tag_decoder",
      //   task: "feats",
      //   encoder: pass_through_layer,
      //   adaptive: true
      //   dropout: std_dropout,
      // },
      // lemmas: {
      //   type: "udify_tag_decoder",
      //   task: "lemmas",
      //   encoder: pass_through_layer,
      //   adaptive: true,
      //   dropout: std_dropout,
      // },
      deps: {
        type: "udify_dependency_decoder",
        encoder: pass_through_layer,
        tag_representation_dim: 256,
        arc_representation_dim: 768,
        dropout: std_dropout,
      }
    }
  },

  trainer: {   
    cuda_device: 0,
    num_epochs: num_epochs,
    patience: patience,
    validation_metric: "+.run/.sum",
    num_gradient_accumulation_steps: 1,
    checkpointer: {
      keep_most_recent_by_count: 1,
      save_every_num_seconds: 1*60*60 // every hour
    },
    grad_norm: 5,
    grad_clipping: 10,
    run_confidence_checks: false,
    callbacks: [
      { 
        type: "wandb",
        summary_interval: 50,
        should_log_parameter_statistics: true,
        should_log_learning_rate: true,
        project: "",
        entity: "tomeffland",
        watch_model: true,
        name: wandb_name,
        notes: wandb_note,
        tags: [wandb_tag],
      },
    ],

    optimizer: {
        type: "huggingface_adamw",
        lr: learning_rate,
        betas: [0.9, 0.99],
        weight_decay: weight_decay,
    },
    learning_rate_scheduler: {
      type: "linear_with_warmup",
      num_epochs: scheduler_num_epochs,
      warmup_steps: 500,
      num_steps_per_epoch: num_batches_per_epoch,
    },
  }
}