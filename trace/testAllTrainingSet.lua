--[[

  Training script for semantic relatedness prediction on the SICK dataset.

--]]

require('..')

local args = lapp [[
  -m,--model  (default bigru)        Model architecture: [lstm, bilstm, averagevect]
]]

local model_dir = tracenn.models_dir
local model_file_name = 'trainingDataOnly_1.model'

header('Test trained model:')
if args.model ==  'averagevect' then
  model = tracenn.AverageVectTrace.load(model_dir .. model_file_name)
else
  model = tracenn.RNNTrace.load(model_dir .. model_file_name)
end

-- directory containing dataset files
local data_dir = tracenn.data_dir ..'/trace_All_For_Training/'
local artifact_dir = tracenn.artifact_dir
-- load artifact vocab
local vocab = tracenn.Vocab(artifact_dir .. 'vocab_ptc_artifact_clean.txt')
-- load all artifact
local artifact = tracenn.read_artifact(artifact_dir, vocab)

-- Map artifact to word embeddings
for i = 1, #artifact.src_artfs do
  local src_artf = artifact.src_artfs[i]
  artifact.src_artfs[i] = model.emb_vecs:index(1, src_artf:long())
end

for i = 1, #artifact.trg_artfs do
  local trg_artf = artifact.trg_artfs[i]
  artifact.trg_artfs[i] = model.emb_vecs:index(1, trg_artf:long())
end

local training_dir = data_dir .. 'train/'
header('Reading all training data')
local train_dataset = tracenn.read_trace_dataset(training_dir, vocab)
header('Evaluating on all training data')
local train_loss, train_predictions = model:predict_dataset(train_dataset, artifact)

print('Done with Train loss:', train_loss)

local file_idx = 1
local predictions_save_path
while true do
  predictions_save_path = string.format(
    tracenn.predictions_dir .. '/' .. model_file_name ..'.pred')
  -- check if the files already exist in the folder.
  if lfs.attributes(predictions_save_path) == nil then
    break
  end
  file_idx = file_idx + 1
end

-- write predictions to disk
local predictions_file = torch.DiskFile(predictions_save_path, 'w')
predictions_file:noAutoSpacing()
print('writing predictions to ' .. predictions_save_path)
for i = 1, #train_predictions do
  if args.model == 'averagevect' then
    for j = 1, train_predictions[i]:size(2) do
      predictions_file:writeDouble(train_predictions[i][1][j])
      predictions_file:writeString(',')
    end
  else
    for j = 1, train_predictions[i]:size(1) do
      predictions_file:writeDouble(train_predictions[i][j])
      predictions_file:writeString(',')
    end
  end
  predictions_file:writeString('\n')
end
predictions_file:close()
