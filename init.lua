require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')
-- require('cutorch')
-- require('cunn')

tracenn = {}

include('util/read_data.lua')
include('util/Vocab.lua')
include('models/IdentityLinear.lua')
include('models/LSTM.lua')
include('models/GRU.lua')
include('models/IRNN.lua')
include('trace/RNNTrace.lua')
include('trace/AverageVectTrace.lua')


printf = utils.printf

-- global paths (modify if desired)
tracenn.output = '/Users/Jinguo/Dropbox/TraceNN_experiment/tracenn/'
tracenn.data_dir        = tracenn.output .. 'data/'
tracenn.models_dir      = tracenn.output .. 'trained_models/'
tracenn.predictions_dir = tracenn.output .. 'predictions/'
tracenn.artifact_dir = tracenn.data_dir .. 'artifact/'

-- share module parameters
function share_params(cell, src)
  if torch.type(cell) == 'nn.gModule' then
    for i = 1, #cell.forwardnodes do
      local node = cell.forwardnodes[i]
      if node.data.module then
        node.data.module:share(src.forwardnodes[i].data.module,
          'weight', 'bias', 'gradWeight', 'gradBias')
      end
    end
  elseif torch.isTypeOf(cell, 'nn.Module') then
    cell:share(src, 'weight', 'bias', 'gradWeight', 'gradBias')
  else
    error('parameters cannot be shared for this input')
  end
end

function header(s)
  print(string.rep('-', 80))
  print(s)
  print(string.rep('-', 80))
end

function string.starts(String,Start)
   return string.sub(String,1,string.len(Start))==Start
end
