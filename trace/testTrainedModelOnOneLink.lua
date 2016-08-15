--[[

  Training script for semantic relatedness prediction on the SICK dataset.

--]]

require('..')

local args = lapp [[
  -m,--model  (default gru)        Model architecture: [lstm, bilstm, averagevect]
]]

local model_dir = tracenn.models_dir
local model_file_name = '1470992031.4712.model'

header('Test trained model:')
if args.model ==  'averagevect' then
  model = tracenn.AverageVectTrace.load(model_dir .. model_file_name)
else
  model = tracenn.RNNTrace.load(model_dir .. model_file_name)
end

-- print information
model:print_config()

local source_string = 'the bos software design shall provide mechanisms to minimize the data corruption of stored data in the bos  </s> '
local target_string = 'the steps of the above sequence are as follows  the onboard sends a 02020 poll registration message to the bos  </s> '
  ..' the bos validates the received 02020 message as described in pii ssdd 2013  </s> '
  -- ..' the poll handler then forks two threads to process the message  the poll handler  ph  thread and the mandatory directive checker  mdc  thread  </s> '
  --  ..'both threads retrieve the opk from the tm_locomotive datapoint using the train library  </s> '
  --  .. 'please note that not diversifying the implementation of this step  i e   relying on the implementation of the train library instead  is acceptable since the message validation was already done once using a different approach  as depicted in pii ssdd 2014  </s> '
  --  .. 'both threads validate the message hmac using the retrieved opk  </s> '
  --  .. 'if either validation fails  an ob_badhmac error is triggered and thread processing stops  </s> '
  -- .. 'upon hmac validation success  the ph thread performs the payload validation of the 02020 message as described in pii ssdd 2770  </s> '
  --  .. 'the threads continue with update subdivision polling list functionality  </s>'

  -- target_string = 'if the header is successfully validated  the messaging library performs the payload validation  the only payload validation done here is the validation of the railroad scac and may result adding the ob_badscac error code to the validation results   </s> '

function readInput(input, vocab)
  if input == nil then return end
  local tokens = stringx.split(input)
  local len = #tokens
  local sent = {}
  for i = 1, len do
    local token = tokens[i]
    if vocab:contains(token) then
      sent[i] = vocab:index(token)
    end
  end
  return torch.IntTensor(sent)
end

-- load the vocab for trace data
-- directory containing dataset files
local data_dir = tracenn.data_dir ..'/trace_all/'
local artifact_dir = tracenn.artifact_dir
-- load artifact vocab
local vocab = tracenn.Vocab(artifact_dir .. 'vocab_ptc_artifact_clean.txt')
-- load all artifact
-- local artifact = tracenn.read_artifact(artifact_dir, vocab)

-- convert sentence string to the index in the vocab list
local lsent = readInput(source_string, vocab)
local rsent = readInput(target_string, vocab)

local output = model:predict_text(lsent, rsent)
print('Output:',output:exp())

print(model.sim_module)
print(model.sim_module:get(2).weight)
print(model.sim_module:get(2).bias)
