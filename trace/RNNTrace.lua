--[[

  Trace using RNN models.

--]]

local Trace = torch.class('tracenn.RNNTrace')

function Trace:__init(config)
  self.hidden_dim    = config.hidden_dim    or 50
  self.learning_rate = config.learning_rate or 0.0001
  self.batch_size    = config.batch_size    or 10
  self.num_layers    = config.num_layers    or 1
  self.reg           = config.reg           or 0
  self.structure     = config.structure     or 'lstm'
  self.sim_nhidden   = config.sim_nhidden   or 10
  self.grad_clip     = config.grad_clip     or 10

  -- word embedding
  self.emb_vecs = config.emb_vecs
  self.emb_dim = config.emb_vecs:size(2)

  -- number of similarity rating classes
  self.num_classes = 2
  self.class_weight = torch.Tensor({1, 20})

  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate }

  -- Set Objective as minimize Negative Log Likelihood
  -- Remember to set the size_average to false to use the effect of weight!!
  -- self.criterion = nn.ClassNLLCriterion(self.class_weight, false)
  self.criterion = nn.ClassNLLCriterion()

  -- initialize RNN model
  local rnn_config = {
    in_dim = self.emb_dim,
    hidden_dim = self.hidden_dim,
    num_layers = self.num_layers,
    gate_output = true, -- ignored by RNN models other than LSTM
  }

  if self.structure == 'lstm' then
    self.lrnn = tracenn.LSTM(rnn_config) -- "left" RNN
    self.rrnn = tracenn.LSTM(rnn_config) -- "right" RNN
  elseif self.structure == 'bilstm' then
    self.lrnn = tracenn.LSTM(rnn_config)
    self.lrnn_b = tracenn.LSTM(rnn_config) -- backward "left" RNN
    self.rrnn = tracenn.LSTM(rnn_config)
    self.rrnn_b = tracenn.LSTM(rnn_config) -- backward "right" RNN
  elseif self.structure == 'irnn' then
    self.lrnn = tracenn.IRNN(rnn_config) -- "left" RNN
    self.rrnn = tracenn.IRNN(rnn_config) -- "right" RNN
  elseif self.structure == 'birnn' then
    self.lrnn = tracenn.IRNN(rnn_config)
    self.lrnn_b = tracenn.IRNN(rnn_config) -- backward "left" RNN
    self.rrnn = tracenn.IRNN(rnn_config)
    self.rrnn_b = tracenn.IRNN(rnn_config) -- backward "right" RNN
  elseif self.structure == 'gru' then
    self.lrnn = tracenn.GRU(rnn_config) -- "left" RNN
    self.rrnn = tracenn.GRU(rnn_config) -- "right" RNN
  elseif self.structure == 'bigru' then
    self.lrnn = tracenn.GRU(rnn_config)
    self.lrnn_b = tracenn.GRU(rnn_config) -- backward "left" RNN
    self.rrnn = tracenn.GRU(rnn_config)
    self.rrnn_b = tracenn.GRU(rnn_config) -- backward "right" RNN
  else
    error('invalid RNN type: ' .. self.structure)
  end

  -- similarity model
  self.sim_module = self:new_sim_module()
  local modules = nn.Parallel()
    :add(self.lrnn)
    :add(self.sim_module)
  self.params, self.grad_params = modules:getParameters()
  self.rnn_params = self.lrnn:parameters()
  self.rnn_params_element_number = 0
  for i=1,#self.rnn_params do
    self.rnn_params_element_number =
      self.rnn_params_element_number + self.rnn_params[i]:nElement()
  end

  -- print('RNN grad_params',self.rnn_grad_params )

  -- share must only be called after getParameters, since this changes the
  -- location of the parameters
  share_params(self.rrnn, self.lrnn)
  if string.starts(self.structure,'bi') then
    -- tying the forward and backward weights improves performance
    share_params(self.lrnn_b, self.lrnn)
    share_params(self.rrnn_b, self.lrnn)
  end
end

function Trace:new_sim_module()
  local lvec, rvec, inputs, input_dim
  if not string.starts(self.structure,'bi') then
    -- standard (left-to-right) LSTM
    input_dim = 2 * self.num_layers * self.hidden_dim
    local linput, rinput = nn.Identity()(), nn.Identity()()
    if self.num_layers == 1 then
      lvec, rvec = linput, rinput
    else
      lvec, rvec = nn.JoinTable(1)(linput), nn.JoinTable(1)(rinput)
    end
    inputs = {linput, rinput}
  elseif string.starts(self.structure,'bi') then
    -- bidirectional LSTM
    input_dim = 4 * self.num_layers * self.hidden_dim
    local lf, lb, rf, rb = nn.Identity()(), nn.Identity()(), nn.Identity()(), nn.Identity()()
    if self.num_layers == 1 then
      lvec = nn.JoinTable(1){lf, lb}
      rvec = nn.JoinTable(1){rf, rb}
    else
      -- in the multilayer case, each input is a table of hidden vectors (one for each layer)
      lvec = nn.JoinTable(1){nn.JoinTable(1)(lf), nn.JoinTable(1)(lb)}
      rvec = nn.JoinTable(1){nn.JoinTable(1)(rf), nn.JoinTable(1)(rb)}
    end
    inputs = {lf, lb, rf, rb}
  end
  local mult_dist = nn.CMulTable(){lvec, rvec}
  local add_dist = nn.Abs()(nn.CSubTable(){lvec, rvec})
  local vec_dist_feats = nn.JoinTable(1){mult_dist, add_dist}
  local vecs_to_input = nn.gModule(inputs, {vec_dist_feats})

   -- define similarity model architecture
  local sim_module = nn.Sequential()
    :add(vecs_to_input)
    -- :add(nn.Dropout(0.5))
    :add(nn.Linear(input_dim, self.sim_nhidden))
    :add(nn.Sigmoid())    -- does better than tanh
    :add(nn.Linear(self.sim_nhidden, self.num_classes))
    :add(nn.LogSoftMax())
  return sim_module
end

function Trace:train(dataset, artifact)
  self.lrnn:training()
  self.rrnn:training()
  self.sim_module:training()
  if string.starts(self.structure,'bi') then
    self.lrnn_b:training()
    self.rrnn_b:training()
  end

  local indices = torch.randperm(dataset.size)
  local zeros = torch.zeros(self.hidden_dim)
  local train_loss = 0
  for i = 1, dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

    -- get target distributions for batch
    local targets = torch.zeros(batch_size)
    for j = 1, batch_size do
      targets[j] = dataset.labels[indices[i + j - 1]]
    end
    local count = 0

    local feval = function(x)
      if x ~= self.params then
        self.params:copy(x)
      end
      self.grad_params:zero()

      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        local linputs, rinputs

        -- load input artifact content using id
        if artifact.src_artfs_ids[dataset.lsents[idx]]~= nil then
          linputs = artifact.src_artfs[artifact.src_artfs_ids[dataset.lsents[idx]]]
        else
          print('Cannot find source:', dataset.lsents[idx])
          break
        end
        if artifact.trg_artfs_ids[dataset.rsents[idx]]~= nil then
          rinputs = artifact.trg_artfs[artifact.trg_artfs_ids[dataset.rsents[idx]]]
        else
          print('Cannot find target:', rsents[idx])
          break
        end
         -- get sentence representations
        local inputs
        if not string.starts(self.structure,'bi') then
          inputs = {self.lrnn:forward(linputs), self.rrnn:forward(rinputs)}
        elseif  string.starts(self.structure,'bi') then
          inputs = {
            self.lrnn:forward(linputs),
            self.lrnn_b:forward(linputs, true), -- true => reverse
            self.rrnn:forward(rinputs),
            self.rrnn_b:forward(rinputs, true)
          }
        end

        -- compute relatedness
        local output = self.sim_module:forward(inputs)

        -- compute loss and backpropagate
        local example_loss = self.criterion:forward(output, targets[j])

        loss = loss + example_loss
        local sim_grad = self.criterion:backward(output, targets[j])
        local rep_grad = self.sim_module:backward(inputs, sim_grad)
        -- print("Sim grad", sim_grad)

        if not string.starts(self.structure,'bi') then
          local rnn_grad = self:RNN_backward(linputs, rinputs, rep_grad)
          -- print("RNN grad:", rnn_grad)
        elseif  string.starts(self.structure,'bi') then
          self:BiRNN_backward(linputs, rinputs, rep_grad)
        end
      end
      train_loss = train_loss + loss

      loss = loss / batch_size
      -- print('Loss:', loss)
      self.grad_params:div(batch_size)

      -- Gradient clipping: if the norm of rnn gradient is bigger than threshold
      -- scale the gradient to
      -- local sim_params = self.params:narrow(1,self.rnn_params_element_number, self.params:nElement()-self.rnn_params_element_number)
      -- -- print("sim_params:", sim_params)
      -- local rnn_params = self.params:narrow(1,1,self.rnn_params_element_number)
      -- -- print("rnn_params:", rnn_params)
      -- local sim_grad_params = self.grad_params:narrow(1,self.rnn_params_element_number, self.params:nElement()-self.rnn_params_element_number)
      -- -- print("sim_grad_params:", sim_grad_params)
      local rnn_grad_params = self.grad_params:narrow(1,1,self.rnn_params_element_number)
      -- print("rnn_grad_params:", rnn_grad_params)

      local rnn_grad_norm = torch.norm(rnn_grad_params)
      if rnn_grad_norm > self.grad_clip then
        print('clipping gradient')
          rnn_grad_params:div(rnn_grad_norm/self.grad_clip)
      end

      -- regularization
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2 * batch_size/dataset.size
      -- Final derivatives to return after regularization:
      -- self.grad_params + self.reg*self.params
      self.grad_params:add(self.reg, self.params)
      -- count = count + 1
      -- print(count)
      return loss, self.grad_params
    end

  --  print('Check the gradients:', self.grad_params:size(1)*2)
  --  diff, dc, dc_est = optim.checkgrad(feval, self.params:clone())
  --  print('Diff must be close to 1e-8: diff = ' .. diff)

    optim.rmsprop(feval, self.params, self.optim_state)
  end
  train_loss = train_loss/dataset.size
  xlua.progress(dataset.size, dataset.size)
  print('Training loss', train_loss)
  return train_loss
end

-- LSTM backward propagation
function Trace:RNN_backward(linputs, rinputs, rep_grad)
  local lgrad, rgrad
  if self.num_layers == 1 then
    lgrad = torch.zeros(linputs:size(1), self.hidden_dim)
    rgrad = torch.zeros(rinputs:size(1), self.hidden_dim)
    lgrad[linputs:size(1)] = rep_grad[1]
    rgrad[rinputs:size(1)] = rep_grad[2]
  else
    lgrad = torch.zeros(linputs:size(1), self.num_layers, self.hidden_dim)
    rgrad = torch.zeros(rinputs:size(1), self.num_layers, self.hidden_dim)
    for l = 1, self.num_layers do
      lgrad[{linputs:size(1), l, {}}] = rep_grad[1][l]
      rgrad[{rinputs:size(1), l, {}}] = rep_grad[2][l]
    end
  end
  self.lrnn:backward(linputs, lgrad)
  local lstm_grad = self.rrnn:backward(rinputs, rgrad)
  return lstm_grad
end

-- Bidirectional LSTM backward propagation
function Trace:BiRNN_backward(linputs, rinputs, rep_grad)
  local lgrad, lgrad_b, rgrad, rgrad_b
  if self.num_layers == 1 then
    lgrad   = torch.zeros(linputs:size(1), self.hidden_dim)
    lgrad_b = torch.zeros(linputs:size(1), self.hidden_dim)
    rgrad   = torch.zeros(rinputs:size(1), self.hidden_dim)
    rgrad_b = torch.zeros(rinputs:size(1), self.hidden_dim)
    lgrad[linputs:size(1)] = rep_grad[1]
    rgrad[rinputs:size(1)] = rep_grad[3]
    lgrad_b[1] = rep_grad[2]
    rgrad_b[1] = rep_grad[4]
  else
    lgrad   = torch.zeros(linputs:size(1), self.num_layers, self.hidden_dim)
    lgrad_b = torch.zeros(linputs:size(1), self.num_layers, self.hidden_dim)
    rgrad   = torch.zeros(rinputs:size(1), self.num_layers, self.hidden_dim)
    rgrad_b = torch.zeros(rinputs:size(1), self.num_layers, self.hidden_dim)
    for l = 1, self.num_layers do
      lgrad[{linputs:size(1), l, {}}] = rep_grad[1][l]
      rgrad[{rinputs:size(1), l, {}}] = rep_grad[3][l]
      lgrad_b[{1, l, {}}] = rep_grad[2][l]
      rgrad_b[{1, l, {}}] = rep_grad[4][l]
    end
  end
  self.lrnn:backward(linputs, lgrad)
  self.lrnn_b:backward(linputs, lgrad_b, true)
  self.rrnn:backward(rinputs, rgrad)
  self.rrnn_b:backward(rinputs, rgrad_b, true)
end

-- Predict the similarity of a sentence pair (log probability, should use output:exp() for probability).
function Trace:predict(lsent, rsent, artifact)
  self.lrnn:evaluate()
  self.rrnn:evaluate()
  self.sim_module:evaluate()
  local linputs, rinputs
  if artifact.src_artfs_ids[lsent]~= nil then
    linputs = artifact.src_artfs[artifact.src_artfs_ids[lsent]]
  else
    print('Cannot find source:', lsent)
    return nil
  end
  if artifact.trg_artfs_ids[rsent]~= nil then
    rinputs = artifact.trg_artfs[artifact.trg_artfs_ids[rsent]]
  else
    print('Cannot find target:', rsent)
    return nil
  end
  local inputs
  if not  string.starts(self.structure,'bi') then
    inputs = {self.lrnn:forward(linputs), self.rrnn:forward(rinputs)}
  elseif  string.starts(self.structure,'bi') then
    self.lrnn_b:evaluate()
    self.rrnn_b:evaluate()
    inputs = {
      self.lrnn:forward(linputs),
      self.lrnn_b:forward(linputs, true),
      self.rrnn:forward(rinputs),
      self.rrnn_b:forward(rinputs, true)
    }
  end
  local output = self.sim_module:forward(inputs)
  self.lrnn:forget()
  self.rrnn:forget()
  if  string.starts(self.structure,'bi') then
    self.lrnn_b:forget()
    self.rrnn_b:forget()
  end
  -- Jin: bug in original code. range is changed to [1, # of classes]
--  return torch.range(1, self.num_classes):dot(output:exp())
  return output
end

-- Produce similarity predictions for each sentence pair in the dataset.
function Trace:predict_dataset(dataset, artifact)
  local predictions = {}
  local targets = dataset.labels
  local loss = 0
  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    local lsent, rsent = dataset.lsents[i], dataset.rsents[i]
    local output = self:predict(lsent, rsent, artifact)
    predictions[i] = torch.exp(output)
    local example_loss = self.criterion:forward(output, targets[i])
    loss = loss + example_loss
  end
  loss = loss/dataset.size
  return loss, predictions
end

function Trace:compute_loss_dataset(dataset, artifact)
  local targets = dataset.labels
  local loss = 0
  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    local lsent, rsent = dataset.lsents[i], dataset.rsents[i]
    local output = self:predict(lsent, rsent, artifact)
    local example_loss = self.criterion:forward(output, targets[i])
    loss = loss + example_loss
  end
  loss = loss/dataset.size
  return loss
end

function Trace:print_config()
  local num_params = self.params:nElement()
  local num_sim_params = self:new_sim_module():getParameters():nElement()
  printf('%-25s = %d\n',   'num params', num_params)
  printf('%-25s = %d\n',   'num compositional params', num_params - num_sim_params)
  printf('%-25s = %d\n',   'word vector dim', self.emb_dim)
  printf('%-25s = %d\n',   'RNN hidden dim', self.hidden_dim)
  printf('%-25s = %.2e\n', 'regularization strength', self.reg)
  printf('%-25s = %d\n',   'minibatch size', self.batch_size)
  printf('%-25s = %.2e\n', 'learning rate', self.learning_rate)
  printf('%-25s = %s\n',   'RNN structure', self.structure)
  printf('%-25s = %d\n',   'RNN layers', self.num_layers)
  printf('%-25s = %d\n',   'sim module hidden dim', self.sim_nhidden)
  printf('%-25s = %d\n',   'Gradient clip', self.grad_clip)
end

--
-- Serialization
--

function Trace:save(path)
  local config = {
    batch_size    = self.batch_size,
    emb_vecs      = self.emb_vecs,
    learning_rate = self.learning_rate,
    num_layers    = self.num_layers,
    hidden_dim    = self.hidden_dim,
    sim_nhidden   = self.sim_nhidden,
    reg           = self.reg,
    structure     = self.structure,
    grad_clip     = self.grad_clip
  }

  torch.save(path, {
    params = self.params,
    config = config,
  })
end

function Trace.load(path)
  local state = torch.load(path)
  local model = tracenn.RNNTrace.new(state.config)
  model.params:copy(state.params)
  return model
end
