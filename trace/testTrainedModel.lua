--[[

  Training script for semantic relatedness prediction on the SICK dataset.

--]]

require('..')

-- read command line arguments
local args = lapp [[
Training script for semantic relatedness prediction on the TRACE dataset.
  -m,--model  (default lstm) Model architecture: [lstm, bilstm]
  -l,--layers (default 1)          Number of layers (ignored for Tree-LSTM)
  -d,--dim    (default 30)        LSTM memory dimension
  -e,--epochs (default 50)         Number of training epochs
]]

local model_dir = 'trained_models/'
local model_file_name = 'rel-lstm.2l.10d.1.th'

header('Test trained model:')
model = treelstm.LSTMSim.load(model_dir .. model_file_name)

local source_string = 'the bos shall set the received message type field in an application acknowledgement   mbbsackb    or mbciackb      message to the message type of the received dispatching system message being acknowledged'
local target_string = 'the steps in the above diagram are as follows   a bos application component calls the sendapplicationacknowledgment function of the messaging library with a list of response codes   the type and sequence number of the message being acknowledged and the subdivision id'

function readInput(input, vocab)
  if input == nil then return end
  local tokens = stringx.split(input)
  local len = #tokens
  local sent = torch.IntTensor(len)
  for i = 1, len do
    local token = tokens[i]
    sent[i] = vocab:index(token)
  end
  return sent
end

-- load the vocab for trace data
local data_dir = 'data/trace_balanced/'
local vocab = treelstm.Vocab(data_dir .. 'vocab-cased.txt')
-- convert sentence string to the index in the vocab list
local lsent = readInput(source_string, vocab)
local rsent = readInput(target_string, vocab)

local output = model:predict(lsent, rsent)
print('Output:',output)
