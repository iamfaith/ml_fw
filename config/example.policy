framework: 'keras' #torch

project: 'image-classification'

base_path: '/content/gdrive/My Drive/dataset'  #/google drive

# wandb: True
train:
  # dir: './results/resnet34.0.policy'
  batch_size: 16
  epochs: 32
  learning_rate: 0.0001

transform:
  name: 'default'
  rules: './transform/rules.json'
  params:
    per_image_norm: True
    size: 299

optimizer:
  name: 'adam'
  params:
    lr: 0.0005

data:
  num_workers: 4