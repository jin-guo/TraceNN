--[[

 Recurrent Neural Network.

--]]

local IRNN, parent = torch.class('tracenn.IRNN', 'nn.Module')

function IRNN:__init(config)
  parent.__init(self)

  self.in_dim = config.in_dim
  self.hidden_dim = config.hidden_dim or 50
  self.num_layers = config.num_layers or 1

  self.master_cell = self:new_cell()
  self.depth = 0
  self.cells = {}  -- table of cells in a roll-out

  -- initial (t = 0) states for forward propagation and initial error signals
  -- for backpropagation
  local otable_init, otable_grad, htable_init, htable_grad
  if self.num_layers == 1 then
    otable_init = torch.zeros(self.hidden_dim)
    htable_init = torch.zeros(self.hidden_dim)
    otable_grad = torch.zeros(self.hidden_dim)
    htable_grad = torch.zeros(self.hidden_dim)
  else
    otable_init, otable_grad, htable_init, htable_grad = {}, {}, {}, {}
    for i = 1, self.num_layers do
      otable_init[i] = torch.zeros(self.hidden_dim)
      htable_init[i] = torch.zeros(self.hidden_dim)
      otable_grad[i] = torch.zeros(self.hidden_dim)
      htable_grad[i] = torch.zeros(self.hidden_dim)
    end
  end
  self.initial_values = {otable_init, htable_init}
  self.gradInput = {
    torch.zeros(self.in_dim),
    htable_grad
  }
end

-- Instantiate a new RNN cell.
-- Each cell shares the same parameters, but the activations of their constituent
-- layers differ.
function IRNN:new_cell()
  local input = nn.Identity()()
  local htable_p = nn.Identity()()

  -- multilayer RNN
  local htable = {}
  local otable = {}
  for layer = 1, self.num_layers do
    local h_p = (self.num_layers == 1) and htable_p or nn.SelectTable(layer)(htable_p)

    local in_module = (layer == 1)
      and nn.Linear(self.in_dim, self.hidden_dim)(input)
      or  nn.Linear(self.hidden_dim, self.hidden_dim)(htable[layer - 1])
    htable[layer] = nn.ReLU()(nn.CAddTable(){
      in_module,
      nn.IdentityLinear(self.hidden_dim, self.hidden_dim)(h_p)
    })
    otable[layer] = nn.Linear(self.hidden_dim, self.hidden_dim)(htable[layer])
  end

  -- if RNN is single-layered, this makes htable/ctable Tensors (instead of tables).
  -- this avoids some quirks with nngraph involving tables of size 1.
  otable = nn.Identity()(otable)
  htable = nn.Identity()(htable)
  local cell = nn.gModule({input, htable_p}, {otable, htable})

  -- share parameters
  if self.master_cell then
    share_params(cell, self.master_cell)
  end
  return cell
end

-- Forward propagate.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- reverse: if true, read the input from right to left (useful for bidirectional LSTMs).
-- Returns the final hidden state of the LSTM.
function IRNN:forward(inputs, reverse)
  local size = inputs:size(1)
  for t = 1, size do
    local input = reverse and inputs[size - t + 1] or inputs[t]
    self.depth = self.depth + 1
    local cell = self.cells[self.depth]
    if cell == nil then
      cell = self:new_cell()
      self.cells[self.depth] = cell
    end
    local prev_output
    if self.depth > 1 then
      prev_output = self.cells[self.depth - 1].output
    else
      prev_output = self.initial_values
    end

    local outputs = cell:forward({input, prev_output[2]})
    local otable, htable = unpack(outputs)
    if self.num_layers == 1 then
      self.output = otable
    else
      self.output = {}
      for i = 1, self.num_layers do
        self.output[i] = otable[i]
      end
    end
  end
  return self.output
end

-- Backpropagate. forward() must have been called previously on the same input.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- grad_outputs: T x num_layers x mem_dim tensor.
-- reverse: if true, read the input from right to left.
-- Returns the gradients with respect to the inputs (in the same order as the inputs).
function IRNN:backward(inputs, grad_outputs, reverse)
  local size = inputs:size(1)
  if self.depth == 0 then
    error("No cells to backpropagate through")
  end

  local input_grads = torch.Tensor(inputs:size())
  for t = size, 1, -1 do
    local input = reverse and inputs[size - t + 1] or inputs[t]
    local grad_output = reverse and grad_outputs[size - t + 1] or grad_outputs[t]
    local cell = self.cells[self.depth]
    local grads = {grad_output, self.gradInput[2]}
    

    local prev_output = (self.depth > 1) and self.cells[self.depth - 1].output
                                         or self.initial_values
    self.gradInput = cell:backward({input, prev_output[2]}, grads)
    if reverse then
      input_grads[size - t + 1] = self.gradInput[1]
    else
      input_grads[t] = self.gradInput[1]
    end
    self.depth = self.depth - 1
  end
  self:forget() -- important to clear out state
  return input_grads
end

function IRNN:share(rnn, ...)
  if self.in_dim ~= rnn.in_dim then error("RNN input dimension mismatch") end
  if self.hidden_dim ~= rnn.hidden_dim then error("RNN hidden dimension mismatch") end
  if self.num_layers ~= rnn.num_layers then error("RNN layer count mismatch") end
  share_params(self.master_cell, rnn.master_cell, ...)
end

function IRNN:zeroGradParameters()
  self.master_cell:zeroGradParameters()
end

function IRNN:parameters()
  return self.master_cell:parameters()
end

-- Clear saved gradients
function IRNN:forget()
  self.depth = 0
  for i = 1, #self.gradInput do
    local gradInput = self.gradInput[i]
    if type(gradInput) == 'table' then
      for _, t in pairs(gradInput) do t:zero() end
    else
      self.gradInput[i]:zero()
    end
  end
end
