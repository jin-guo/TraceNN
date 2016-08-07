--[[

  Training script for semantic relatedness prediction on the SICK dataset.

--]]

require('..')

local args = lapp [[
  -m,--model  (default averagevect)        Model architecture: [lstm, bilstm, averagevect]
]]

local model_dir = tracenn.models_dir
local model_file_name = '/rel-averagevect.2l.30d.12.th'

header('Test trained model:')
if args.model ==  'averagevect' then
  model = tracenn.AverageVectTrace.load(model_dir .. model_file_name)
else
  model = tracenn.RNNTrace.load(model_dir .. model_file_name)
end

local source_string = 'the bos shall set the received message type field in an application acknowledgement   mbbsackb    or mbciackb      message to the message type of the received dispatching system message being acknowledged'
local target_string = 'the steps in the above diagram are as follows a bos application component calls the sendapplicationacknowledgment function of the messaging library with a list of response codes   the type and sequence number of the message being acknowledged and the subdivision id'

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
local data_dir = tracenn.data_dir ..'/trace_balanced/'
local vocab = tracenn.Vocab(data_dir .. 'vocab_ptc_artifact_clean_nosymbol.txt')
-- convert sentence string to the index in the vocab list
local lsent = readInput(source_string, vocab)
local rsent = readInput(target_string, vocab)

local output = model:predict(lsent, rsent)
print('Output:',output)
