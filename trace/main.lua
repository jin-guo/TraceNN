--[[

  Training Script for Trace Software Artifacts.

--]]

require('..')

-- Pearson correlation
function pearson(x, y)
  x = x - x:mean()
  y = y - y:mean()
  return x:dot(y) / (x:norm() * y:norm())
end

-- read command line arguments
local args = lapp [[
Training script for semantic relatedness prediction on the TRACE dataset.
  -m,--model  (default averagevect)        Model architecture: [lstm, bilstm, averagevect]
  -l,--layers (default 2)          	Number of layers (ignored for averagevect)
  -d,--dim    (default 30)        	RNN hidden dimension (the same with LSTM memory dim)
  -e,--epochs (default 300)         Number of training epochs
  -s,--s_dim  (default 50)          Number of similairity module hidden dimension
  -r,--learning_rate (default 1.00e-01) Learning Rate during Training NN Model
  -b,--batch_size (default 20)      Batch Size of training data point for each update of parameters
  -c,--grad_clip (default 100)  Gradient clip threshold
]]

local model_name, model_class
if args.model == 'lstm' then
  model_name = 'LSTM'
  model_class = tracenn.RNNTrace
elseif args.model == 'bilstm' then
  model_name = 'Bidirectional LSTM'
  model_class = tracenn.RNNTrace
elseif args.model == 'irnn' then
  model_name = 'IRNN'
  model_class = tracenn.RNNTrace
elseif args.model == 'biirnn' then
  model_name = 'Bidirectional IRNN'
  model_class = tracenn.RNNTrace
elseif args.model == 'gru' then
  model_name = 'GRU'
  model_class = tracenn.RNNTrace
elseif args.model == 'bigru' then
  model_name = 'Bidirectional GRU'
  model_class = tracenn.RNNTrace
elseif args.model == 'averagevect' then
  model_name = 'Average Vector'
  model_class = tracenn.AverageVectTrace
end
local model_structure = args.model
header('Use Model: ' ..model_name .. ' for Tracing')

-- directory containing dataset files
local data_dir = tracenn.data_dir ..'/trace_new/'
-- load artifact vocab
local vocab = tracenn.Vocab(data_dir .. 'vocab_ptc_artifact_clean.txt')
-- load all artifact
local artifact = tracenn.read_artifact(data_dir, vocab)


-- load embeddings
print('Loading word embeddings')
local emb_dir = tracenn.data_dir ..'/wordembedding/'
local emb_prefix = emb_dir .. 'wiki_ptc_nosymbol_100d_w5_i5_vecs'
local emb_vocab, emb_vecs = tracenn.read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.vecs')
local emb_dim
for i, vec in ipairs(emb_vecs) do
  emb_dim = vec:size(1)
  break
end
print('Embedding dim:', emb_dim)

-- use only vectors in vocabulary (not necessary, but gives faster training)
local num_unk = 0
local vecs = torch.Tensor(vocab.size, emb_dim)
for i = 1, vocab.size do
  local w = vocab:token(i)
  if emb_vocab:contains(w) then
    vecs[i] = emb_vecs[emb_vocab:index(w)]
  else
    print(w)
    num_unk = num_unk + 1
    vecs[i]:uniform(-0.05, 0.05)
  end
end
print('unk count = ' .. num_unk)
emb_vocab = nil
emb_vecs = nil
collectgarbage()

-- map artifact to word embeddings
for i = 1, #artifact.src_artfs do
  local src_artf = artifact.src_artfs[i]
  artifact.src_artfs[i] = vecs:index(1, src_artf:long())
end

for i = 1, #artifact.trg_artfs do
  local src_artf = artifact.trg_artfs[i]
  artifact.trg_artfs[i] = vecs:index(1, src_artf:long())
end

-- load datasets
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

-- initialize model
local model = model_class{
  emb_vecs   = vecs,
  structure  = model_structure,
  num_layers = args.layers,
  hidden_dim  = args.dim,
  sim_nhidden = args.s_dim,
  learning_rate = args.learning_rate,
  batch_size = args.batch_size,
  grad_clip = args.grad_clip,
}

-- number of epochs to train
local num_epochs = args.epochs

-- print information
header('model configuration')
printf('max epochs = %d\n', num_epochs)
model:print_config()

-- train
local train_start = sys.clock()
local best_dev_loss = 100000000
local last_dev_loss = 100000000
local best_dev_model = model
header('Start Training model')
for i = 1, num_epochs do
  local start = sys.clock()
  printf('-- epoch %d\n', i)
  model:train(train_dataset, artifact)
  printf('-- finished epoch in %.2fs\n', sys.clock() - start)

  -- uncomment to compute train scores
  --[[
  local train_predictions = model:predict_dataset(train_dataset)
  local train_score = pearson(train_predictions, train_dataset.labels)
  printf('-- train score: %.4f\n', train_score)
  --]]

  local dev_loss = model:compute_loss_dataset(dev_dataset, artifact)
  printf('-- dev loss: %.4f\n', dev_loss)

  if dev_loss < best_dev_loss then
    best_dev_loss = dev_loss
    best_dev_model = model_class{
      emb_vecs = vecs,
      structure = model_structure,
      num_layers = args.layers,
      hidden_dim    = args.dim,
      sim_nhidden = args.s_dim,
      learning_rate = args.learning_rate,
      batch_size = args.batch_size,
      grad_clip = args.grad_clip,
    }
    best_dev_model.params:copy(model.params)
  end

  if(dev_loss > last_dev_loss and i>50 and model.learning_rate > 1e-8) then
    model.learning_rate = model.learning_rate/2
    print("Learning rate changed to:", model.learning_rate)
  end
  last_dev_loss = dev_loss
end
printf('finished training in %.2fs\n', sys.clock() - train_start)

-- evaluate
header('Evaluating on test set')
printf('-- using model with dev score = %.4f\n', best_dev_loss)
local test_predictions = best_dev_model:predict_dataset(test_dataset, artifact)
local test_loss = best_dev_model:compute_loss_dataset(test_dataset, artifact)
printf('-- test loss: %.4f\n', test_loss)

-- create predictions and model directories if necessary
if lfs.attributes(tracenn.predictions_dir) == nil then
  lfs.mkdir(tracenn.predictions_dir)
end

if lfs.attributes(tracenn.models_dir) == nil then
  lfs.mkdir(tracenn.models_dir)
end

-- get paths
local file_idx = 1
local predictions_save_path, model_save_path
while true do
  predictions_save_path = string.format(
    tracenn.predictions_dir .. '/rel-%s.%dl.%dd.%d.pred', args.model, args.layers, args.dim, file_idx)
  model_save_path = string.format(
    tracenn.models_dir .. '/rel-%s.%dl.%dd.%d.th', args.model, args.layers, args.dim, file_idx)
  -- check if the files already exist in the folder.
  if lfs.attributes(predictions_save_path) == nil and lfs.attributes(model_save_path) == nil then
    break
  end
  file_idx = file_idx + 1
end

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

-- write models to disk
print('writing model to ' .. model_save_path)
best_dev_model:save(model_save_path)

-- to load a saved model
-- local loaded = model_class.load(model_save_path)
