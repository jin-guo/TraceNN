--[[

 Long Short-Term Memory.

--]]

local GRU, parent = torch.class('tracenn.GRU', 'nn.Module')

function GRU:__init(config)
  parent.__init(self)

  self.in_dim = config.in_dim
  self.hidden_dim = config.hidden_dim or 50
  self.num_layers = config.num_layers or 1
  self.gate_output = config.gate_output
  if self.gate_output == nil then self.gate_output = true end

  self.master_cell = self:new_cell()
  self.depth = 0
  self.cells = {}  -- table of cells in a roll-out

  -- initial (t = 0) states for forward propagation and initial error signals
  -- for backpropagation
  local htable_init, htable_grad
  if self.num_layers == 1 then
    htable_init = torch.zeros(self.hidden_dim)
    htable_grad = torch.zeros(self.hidden_dim)
  else
    htable_init, htable_grad = {}, {}
    for i = 1, self.num_layers do
      htable_init[i] = torch.zeros(self.hidden_dim)
      htable_grad[i] = torch.zeros(self.hidden_dim)
    end
  end
  self.initial_values = htable_init
  self.gradInput = {
    torch.zeros(self.in_dim),
    htable_grad
  }
end

-- Instantiate a new GRU cell.
-- Each cell shares the same parameters, but the activations of their constituent
-- layers differ.
function GRU:new_cell()
  local input = nn.Identity()()
  local htable_p = nn.Identity()()

  -- multilayer GRU
  local htable = {}
  for layer = 1, self.num_layers do
    local h_p = (self.num_layers == 1) and htable_p or nn.SelectTable(layer)(htable_p)

    local new_gate = function()
      local in_module = (layer == 1)
        and nn.Linear(self.in_dim, self.hidden_dim)(input)
        or  nn.Linear(self.hidden_dim, self.hidden_dim)(htable[layer - 1])
      return nn.CAddTable(){
        in_module,
        nn.Linear(self.hidden_dim, self.hidden_dim)(h_p)
      }
    end

    -- update, and reset gates
    local z = nn.Sigmoid()(new_gate())
    local r = nn.Sigmoid()(new_gate())

    local in_module = (layer == 1)
      and nn.Linear(self.in_dim, self.hidden_dim)(input)
      or  nn.Linear(self.hidden_dim, self.hidden_dim)(htable[layer - 1])

    local h_candidate = nn.Tanh()(nn.CAddTable(){
      in_module,
      nn.Linear(self.hidden_dim, self.hidden_dim)(nn.CMulTable(){r, h_p})
    })

    local interposition_part_one =
      nn.CMulTable(){
        nn.AddConstant(1,false)(nn.MulConstant(-1,false)(z)),
        h_p
      }
    local interposition_part_two =
      nn.CMulTable(){
        z, h_candidate
      }

    htable[layer] = nn.CAddTable(){interposition_part_one, interposition_part_two}
  end

  -- if GRU is single-layered, this makes htable/ctable Tensors (instead of tables).
  -- this avoids some quirks with nngraph involving tables of size 1.
  htable = nn.Identity()(htable)
  local cell = nn.gModule({input, htable_p}, {htable})

  -- share parameters
  if self.master_cell then
    share_params(cell, self.master_cell)
  end
  return cell
end

-- Forward propagate.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- reverse: if true, read the input from right to left (useful for bidirectional GRUs).
-- Returns the final hidden state of the GRU.
function GRU:forward(inputs, reverse)
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

    local htable = cell:forward({input, prev_output})
    if self.num_layers == 1 then
      self.output = htable
    else
      self.output = {}
      for i = 1, self.num_layers do
        self.output[i] = htable[i]
      end
    end
  end
  return self.output
end

-- Backpropagate. forward() must have been called previously on the same input.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- grad_outputs: T x num_layers x hidden_dim tensor.
-- reverse: if true, read the input from right to left.
-- Returns the gradients with respect to the inputs (in the same order as the inputs).
function GRU:backward(inputs, grad_outputs, reverse)
  local size = inputs:size(1)
  if self.depth == 0 then
    error("No cells to backpropagate through")
  end

  local input_grads = torch.Tensor(inputs:size())
  for t = size, 1, -1 do
    local input = reverse and inputs[size - t + 1] or inputs[t]
    local grad_output = reverse and grad_outputs[size - t + 1] or grad_outputs[t]
    local cell = self.cells[self.depth]
    local grads = self.gradInput[2]
    if self.num_layers == 1 then
      grads:add(grad_output)
    else
      for i = 1, self.num_layers do
        grads[i]:add(grad_output[i])
      end
    end

    local prev_output = (self.depth > 1) and self.cells[self.depth - 1].output
                                         or self.initial_values
    self.gradInput = cell:backward({input, prev_output}, grads)
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

function GRU:share(GRU, ...)
  if self.in_dim ~= GRU.in_dim then error("GRU input dimension mismatch") end
  if self.hidden_dim ~= GRU.hidden_dim then error("GRU memory dimension mismatch") end
  if self.num_layers ~= GRU.num_layers then error("GRU layer count mismatch") end
  if self.gate_output ~= GRU.gate_output then error("GRU output gating mismatch") end
  share_params(self.master_cell, GRU.master_cell, ...)
end

function GRU:zeroGradParameters()
  self.master_cell:zeroGradParameters()
end

function GRU:parameters()
  return self.master_cell:parameters()
end

-- Clear saved gradients
function GRU:forget()
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
