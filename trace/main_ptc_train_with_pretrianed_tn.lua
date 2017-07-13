
require('..')

-- read command line arguments
local args = lapp [[
Training script for semantic relatedness prediction on the TRACE dataset.
  -e,--epochs (default 40)                 Number of training epochs
  -r,--learning_rate (default 1.00e-03)    Learning Rate during Training NN Model
  -b,--batch_size (default 1)              Batch Size of training data point for each update of parameters
  -c,--grad_clip (default 10)              Gradient clip threshold
  -t,--test_model (default 0)             test model on the testing data
  -g,--reg  (default 1.00e-04)             Regulation lamda
  -o,--output_dir (default '/Users/Jinguo/Dropbox/TraceNN_experiment/tse/') Output directory
  -p,--progress_output (default 'ptc_test_trained_ptc_big_model') Name of the progress output file
  -u,--update_word_emb (default 0)         Update wordEmbedding flag
  -n,--negative_sampling_ratio (default 10) The ratio of negative links to positive links during each epoch
  -m,--model_file_name (default 'ptc_small_n10_a9.model')            The name of the retrained model
]]


-- Update global directories
tracenn.output = args.output_dir
tracenn.data_dir        = tracenn.output .. 'data/'
tracenn.models_dir      = tracenn.output .. 'trained_models/'
tracenn.predictions_dir = tracenn.output .. 'predictions/'
tracenn.progress_dir = tracenn.output .. 'progress/'
tracenn.artifact_dir = tracenn.data_dir .. 'artifact/PTC_small/'

local model_dir = tracenn.models_dir
-- local model_file_name = '1495584415.6021.model'

header('Load Pre-trained Model: '.. args.model_file_name)
local pretrained_model = tracenn.RNNTrace_with_Input_Layer.load(model_dir .. args.model_file_name)

-- local model_name, model_class, model_structure
-- if pretrained_model.structure == 'gru' then
--   model_name = 'GRU'
--   model_class = tracenn.RNNTrace_with_Input_Layer
--   model_structure = 'gru'
-- elseif pretrained_model.structure == 'bigru' then
--   model_name = 'Bidirectional GRU'
--   model_class = tracenn.RNNTrace_with_Input_Layer
--   model_structure = 'bigru'
-- end
-- header('Use Model: ' .. model_name .. ' for Tracing')
--
-- local rnn_in_dim = pretrained_model.emb_dim
-- local rnn_hidden_dim = pretrained_model.hidden_dim
-- local rnn_num_layers = pretrained_model.num_layers
-- local sim_hidden_dim = pretrained_model.sim_nhidden
-- if rnn_in_dim ~= pretrained_model.emb_dim then
--   print('RNN input dimensions do not match')
-- end

-- Initialize model
-- local model = model_class{
--   emb_vecs   = pretrained_model.emb_vecs,
--   structure  = model_structure,
--   num_layers = rnn_num_layers,
--   hidden_dim  = rnn_hidden_dim,
--   sim_nhidden = sim_hidden_dim,
--   learning_rate = args.learning_rate,
--   batch_size = args.batch_size,
--   grad_clip = args.grad_clip,
--   reg = args.reg,
--   update_word_embedding= args.update_word_emb,
-- }
--
-- print('before copying')
-- print(model.lrnn:getParameters())
-- local trained_params = pretrained_model.params
-- model.params:copy(pretrained_model.params)
-- print('after copying')
-- print(model.lrnn:getParameters())

local model = pretrained_model:clone()
model.learning_rate = args.learning_rate
model.batch_size = args.batch_size
model.grad_clip = args.grad_clip
model.reg = args.reg


-- directory containing dataset files
local data_dir = tracenn.data_dir ..'trace_80_10_10_PTC_small_additional_5/'
local artifact_dir = tracenn.artifact_dir
-- load artifact vocab
local vocab = tracenn.Vocab(tracenn.artifact_dir .. 'Vocab.txt')
-- load all artifact
local artifact = tracenn.read_artifact(artifact_dir, vocab)

print('loading datasets')
local train_dir = data_dir .. 'train/'
local dev_dir = data_dir .. 'dev/'
local test_dir = data_dir .. 'test/'
local train_dataset = tracenn.read_trace_dataset(train_dir, vocab)
local dev_dataset = tracenn.read_trace_dataset(dev_dir, vocab)
local test_dataset = tracenn.read_trace_dataset(test_dir, vocab)
printf('num train = %d\n', train_dataset.size)
printf('num dev   = %d\n', dev_dataset.size)
printf('num test  = %d\n', test_dataset.size)

-- Number of epochs to train
local num_epochs = args.epochs

-- print information
header('model configuration')
printf('max epochs = %d\n', num_epochs)
model:print_config()

-- train
local train_start = sys.clock()
local best_dev_loss = 100000000
local last_train_loss = 100000000
local first_train_loss
local best_dev_model = model
-- Save the progress result to tables
local train_loss_progress = {}
local dev_loss_progress = {}
local learning_rate_progress = {}

header('Start Training model')

-- Generate dataset for each epoch contains all link example and
-- "negative_sampling_ratio" times number of random non-link example
local  generate_balanced_dataset = function(dataset_to_balance, ratio)
  local link_index = {}
  local nonlink_index = {}
  local dataset_each_epoch = {}
  dataset_each_epoch.vocab = dataset_to_balance.vocab
  dataset_each_epoch.lsents = {}
  dataset_each_epoch.rsents = {}
  for i = 1, dataset_to_balance.size do
    -- Label 2 indicate trace link, add all tracel ink to dataset
    if dataset_to_balance.labels[i] == 2 then
      link_index[#link_index + 1] = i
      dataset_each_epoch.lsents[#dataset_each_epoch.lsents+1]
        = dataset_to_balance.lsents[i]
      dataset_each_epoch.rsents[#dataset_each_epoch.rsents+1]
        = dataset_to_balance.rsents[i]
    else
      -- Label 1 indicate non trace link, non-link will add randomly within each epoch iteration
      nonlink_index[#nonlink_index + 1] = i
    end
  end
  dataset_each_epoch.size = (1+ratio)*#link_index
  dataset_each_epoch.labels = torch.ones((1+ratio)*#link_index)
  dataset_each_epoch.labels:narrow(1, 1, #link_index):fill(2)
  return dataset_each_epoch, link_index, nonlink_index
end

local neg_ratio = args.negative_sampling_ratio
local training_dataset_each_epoch, training_link_index, training_nolink_index
  = generate_balanced_dataset(train_dataset, neg_ratio)
local dev_dataset_each_epoch, dev_link_index, dev_nolink_index
  = generate_balanced_dataset(dev_dataset, neg_ratio)

for i = 1, num_epochs do

  -- Random select non-link examples for this epoch
  local nonlink_index_selected = torch.randperm(#training_nolink_index)
  for j = 1, neg_ratio*#training_link_index do
    local index = nonlink_index_selected[j]
    training_dataset_each_epoch.lsents[#training_link_index+j]
      = train_dataset.lsents[index]
    training_dataset_each_epoch.rsents[#training_link_index+j]
      = train_dataset.rsents[index]
  end

  local dev_nonlink_index_selected = torch.randperm(#dev_nolink_index)
  for j = 1, neg_ratio*#dev_link_index do
    local index = dev_nonlink_index_selected[j]
    dev_dataset_each_epoch.lsents[#dev_link_index+j]
      = dev_dataset.lsents[index]
    dev_dataset_each_epoch.rsents[#dev_link_index+j]
      = dev_dataset.rsents[index]
  end

  learning_rate_progress[i] = model.learning_rate
  local start = sys.clock()
  printf('-- epoch %d\n', i)
  printf('-- current learning rate %.10f\n', model.learning_rate)
  local train_loss = model:train(training_dataset_each_epoch, artifact)
  printf('-- finished epoch in %.2fs\n', sys.clock() - start)
  printf('-- train loss: %.4f\n', train_loss)


  -- uncomment to compute train scores
  --[[
  local train_predictions = model:predict_dataset(train_dataset)
  local train_score = pearson(train_predictions, train_dataset.labels)
  printf('-- train score: %.4f\n', train_score)
  --]]

  -- local dev_loss = '/'
  -- if i%5 == 0 then
    local dev_loss = model:predict_dataset(dev_dataset_each_epoch, artifact)
    printf('-- dev loss: %.4f\n', dev_loss)

    local to_compare = math.max(train_loss, dev_loss)
    if to_compare < best_dev_loss then
      best_dev_loss = to_compare -- using the max value of training loss and dev loss to select best model
      printf('saving model for best loss in %.2fs\n', best_dev_loss)
      -- best_dev_model = model_class{
      --   emb_vecs = vecs,
      --   structure = model_structure,
      --   num_layers = args.layers,
      --   hidden_dim    = args.dim,
      --   sim_nhidden = args.s_dim,
      --   learning_rate = args.learning_rate,
      --   batch_size = args.batch_size,
      --   grad_clip = args.grad_clip,
      --   update_word_embedding = args.update_word_emb
      -- }
      -- best_dev_model.params:copy(model.params)
      best_dev_model = model:clone()
    end
  -- end
  -- if(train_loss > last_train_loss and model.learning_rate > 1e-8) then
  --   model.learning_rate = model.learning_rate/2
  --   print("Learning rate changed to:", model.learning_rate)
  -- end
  if model.learning_rate > args.learning_rate/100 then
    local alpha = i/500
    model.learning_rate = (1-alpha)*args.learning_rate + alpha*args.learning_rate/100
  end
  last_train_loss = train_loss
  train_loss_progress[i] = train_loss
  dev_loss_progress[i] = dev_loss
  if i == 1 then
    first_train_loss = train_loss
  end
  -- if the loss does not decrease to 80% by epoch 10, stop training
  if i == 20 then
    if train_loss > 0.8*first_train_loss then
      break
    end
  end
end
local training_time = sys.clock() - train_start
printf('finished training in %.2fs\n', training_time)

-- evaluate
header('Evaluating on test set')
printf('-- using model with dev score = %.4f\n', best_dev_loss)
if args.test_model==1 then
  local test_loss, test_predictions = best_dev_model:predict_dataset(test_dataset, artifact)
  printf('-- test loss: %.4f\n', test_loss)

  -- create predictions directories if necessary
  if lfs.attributes(tracenn.predictions_dir) == nil then
    lfs.mkdir(tracenn.predictions_dir)
  end
end

-- create model directories if necessary
if lfs.attributes(tracenn.models_dir) == nil then
  lfs.mkdir(tracenn.models_dir)
end

if lfs.attributes(tracenn.progress_dir) == nil then
  lfs.mkdir(tracenn.progress_dir)
end



-- get paths
local file_idx = 1
local predictions_save_path, model_save_path

-- Save model with the same progress file name
model_save_path = tracenn.models_dir .. args.progress_output ..'.model'

while true do
  predictions_save_path = string.format(
    tracenn.predictions_dir .. 'progress_output', file_idx, '.pred')
  -- model_save_path = string.format(
  --   tracenn.models_dir .. 'rel-%s.%dl.%dd.%d.th', args.model, args.layers, args.dim, file_idx)
  -- check if the files already exist in the folder.
  if lfs.attributes(predictions_save_path) == nil then --and lfs.attributes(model_save_path) == nil then
    break
  end
  file_idx = file_idx + 1
end
if arg.test_model then
  -- write predictions to disk
  local predictions_file = torch.DiskFile(predictions_save_path, 'w')
  predictions_file:noAutoSpacing()
  print('writing predictions to ' .. predictions_save_path)
  for i = 1, #test_predictions do
    if args.model == 'averagevect' then
      for j = 1, test_predictions[i]:size(2) do
        predictions_file:writeDouble(test_predictions[i][1][j])
        predictions_file:writeString(',')
      end
    else
      for j = 1, test_predictions[i]:size(1) do
        predictions_file:writeDouble(test_predictions[i][j])
        predictions_file:writeString(',')
      end
    end
    predictions_file:writeString('\n')
  end
  predictions_file:close()
end

-- write models to disk
print('writing model to ' .. model_save_path)
best_dev_model:save(model_save_path)

local progress_save_path = tracenn.progress_dir .. args.progress_output .. '.txt'
io.output(progress_save_path)
io.write(string.format('Training Duration: %.2fs\n', training_time))
io.write('--------------------------\nModel Configuration:\n--------------------------\n')
io.write(string.format('%-25s = %s\n',   'RNN structure', model.structure))
io.write(string.format('%-25s = %d\n',   'word vector dim', model.emb_dim))
io.write(string.format('%-25s = %d\n',   'update word embedding', model.update_word_embedding))
io.write(string.format('%-25s = %.2e\n', 'regularization strength', model.reg))
io.write(string.format('%-25s = %d\n',   'minibatch size', model.batch_size))
io.write(string.format('%-25s = %.2e\n', 'initial learning rate', args.learning_rate))
io.write(string.format('%-25s = %d\n',   'sim module hidden dim', model.sim_nhidden))
io.write(string.format('%-25s = %d\n',   'negative sampling ratio', args.negative_sampling_ratio))
if model.hidden_dim ~= nil and
  model.num_layers~= nil and
  model.grad_clip~= nil then
  io.write(string.format('%-25s = %d\n',   'RNN hidden dim', model.hidden_dim))
  io.write(string.format('%-25s = %d\n',   'RNN layers', model.num_layers))
  io.write(string.format('%-25s = %d\n',   'Gradient clip', model.grad_clip))
end
io.write('--------------------------\nTraining Progress per epoch:\n--------------------------\n')
io.write('epoch, training_loss,dev_loss,learning_rate\n')
for i = 1, #learning_rate_progress do
  io.write(i, ',')
  io.write(train_loss_progress[i], ',')
  io.write(dev_loss_progress[i], ',')
  io.write(learning_rate_progress[i], '\n')
end
