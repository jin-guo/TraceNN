--[[

  Training script for semantic relatedness prediction on the SICK dataset.

--]]

require('..')

local args = lapp [[
  -m,--model  (default gru)        Model architecture: [lstm, bilstm, averagevect]
]]

local model_dir = tracenn.models_dir .. '/'
local model_file_name = 'rel-gru.2l.30d.2.th'

header('Test trained model:')
if args.model ==  'averagevect' then
  model = tracenn.AverageVectTrace.load(model_dir .. model_file_name)
else
  model = tracenn.RNNTrace.load(model_dir .. model_file_name)
end

-- directory containing dataset files
local data_dir = tracenn.data_dir ..'/trace_all/'
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
  local src_artf = artifact.trg_artfs[i]
  artifact.trg_artfs[i] = model.emb_vecs:index(1, src_artf:long())
end

local test_dir = data_dir .. 'test/'
header('Reading all test data')
local test_dataset = tracenn.read_trace_dataset(test_dir, vocab)
header('Evaluating on test data')
local test_loss, test_predictions = model:predict_dataset(test_dataset, artifact)

print('Done with Test loss:', test_loss)

local file_idx = 1
local predictions_save_path
while true do
  predictions_save_path = string.format(
    tracenn.predictions_dir .. '/' .. model_file_name ..'_OnTestData.pred')
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
