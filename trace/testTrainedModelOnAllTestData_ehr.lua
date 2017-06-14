--[[

  Training script for semantic relatedness prediction on the SICK dataset.

--]]

require('..')

local args = lapp [[
  -m,--model  (default bigru)        Model architecture: [gru, bigru]
  -o,--output_dir (default '/Users/Jinguo/Dropbox/TraceNN_experiment/tse/') Output directory
]]

-- Update global directories
tracenn.output = args.output_dir
tracenn.data_dir        = tracenn.output .. 'data/'
tracenn.models_dir      = tracenn.output .. 'trained_models/'
tracenn.predictions_dir = tracenn.output .. 'predictions/'
tracenn.progress_dir = tracenn.output .. 'progress/'
tracenn.artifact_dir = tracenn.data_dir .. 'artifact/EHR/'

local model_dir = tracenn.models_dir
-- local model_file_name = '1495584415.6021.model'
local model_file_name = 'ehr_autoencoder_trace_r_1eN3_g_1eN4_avg.model'

header('Test trained model:')
model = tracenn.RNNTrace_with_Input_Layer.load(model_dir .. model_file_name)


-- directory containing dataset files
local data_dir = tracenn.data_dir ..'trace_80_10_10_EHR/'
local artifact_dir = tracenn.artifact_dir
-- load artifact vocab
local vocab = tracenn.Vocab(tracenn.artifact_dir .. 'Vocab.txt')
-- load all artifact
local artifact = tracenn.read_artifact(artifact_dir, vocab)



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
    tracenn.predictions_dir .. '/' .. model_file_name ..'best_dev_loss')
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
