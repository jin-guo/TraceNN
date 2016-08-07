--[[

  Training script for semantic relatedness prediction on the SICK dataset.

--]]

require('..')

local args = lapp [[
  -m,--model  (default averagevect)        Model architecture: [lstm, bilstm, averagevect]
]]

local model_dir = tracenn.models_dir .. '/'
local model_file_name = 'rel-averagevect.2l.30d.12.th'

header('Test trained model:')
if args.model ==  'averagevect' then
  model = tracenn.AverageVectTrace.load(model_dir .. model_file_name)
else
  model = tracenn.RNNTrace.load(model_dir .. model_file_name)
end

local data_dir = tracenn.data_dir ..'/trace_balanced/'
local vocab = tracenn.Vocab(data_dir .. 'vocab_ptc_artifact_clean_nosymbol.txt')

local test_dir = data_dir .. 'test_all/'
header('Reading all test data')
local test_dataset = tracenn.read_trace_dataset(test_dir, vocab)
header('Evaluating on all test data')
local test_predictions = model:predict_dataset(test_dataset)

local file_idx = 1
local predictions_save_path
while true do
  predictions_save_path = string.format(
    tracenn.predictions_dir .. '/' .. model_file_name ..'_OnAllTestData.pred')
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
