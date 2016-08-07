--[[

  Trace using Average Vector of All Input Words.

--]]

local AverageVectTrace = torch.class('tracenn.AverageVectTrace')

function AverageVectTrace:__init(config)
  self.learning_rate = config.learning_rate or 0.01
  self.batch_size    = config.batch_size    or 5
  self.reg           = config.reg           or 0
  self.structure     = config.structure     or 'averagevect'
  self.sim_nhidden   = config.sim_nhidden   or 20

  -- word embedding
  self.emb_vecs = config.emb_vecs
  self.emb_dim = config.emb_vecs:size(2)

  -- number of similarity rating classes
  self.num_classes = 2
  self.class_weight = torch.Tensor({1, 500})

  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate }

  -- KL divergence optimization objective
  self.criterion = nn.ClassNLLCriterion(self.class_weight)

  -- similarity model
  self.sim_module = self:new_sim_module()
  self.params, self.grad_params = self.sim_module:getParameters()
end

function AverageVectTrace:new_sim_module()
  local inputs, input_dim

  input_dim = 2 * self.emb_dim
  local linput, rinput = nn.Identity()(), nn.Identity()()
  inputs = {linput, rinput}
  local mult_dist = nn.CMulTable()({linput, rinput})
  local add_dist = nn.Abs()(nn.CSubTable(){linput, rinput})
  local vec_dist_feats = nn.JoinTable(2){mult_dist, add_dist}
  local vecs_to_input = nn.gModule(inputs, {vec_dist_feats})

   -- define similarity model architecture
  local sim_module = nn.Sequential()
    :add(vecs_to_input)
    :add(nn.Linear(input_dim, self.sim_nhidden))
    :add(nn.ReLU())    -- does better than tanh
    :add(nn.Linear(self.sim_nhidden, self.num_classes))
    :add(nn.LogSoftMax())
  return sim_module
end

function AverageVectTrace:train(dataset)
  local indices = torch.randperm(dataset.size)
  local train_loss = 0
  for i = 1, dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

    -- get target label for batch
    local targets = torch.zeros(batch_size)
    for j = 1, batch_size do
      targets[j] = dataset.labels[indices[i + j - 1]]
    end
--    local count = 0
    local feval = function(x)
      if x ~= self.params then
        self.params:copy(x)
      end
      self.grad_params:zero()
      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        local lsent, rsent = dataset.lsents[idx], dataset.rsents[idx]
        local linputs = self.emb_vecs:index(1, lsent:long())
        local rinputs = self.emb_vecs:index(1, rsent:long())


        local linput_aver = torch.sum(linputs, 1):div(linputs:size(1))
        local rinput_aver = torch.sum(rinputs, 1):div(rinputs:size(1))

        -- get sentence representations
        local inputs = {linput_aver, rinput_aver}
        -- compute relatedness
        local output = self.sim_module:forward(inputs)

        -- compute loss and backpropagate
        local example_loss = self.criterion:forward(output, targets[j])
    --    print("Loss:",example_loss)

        loss = loss + example_loss
        local sim_grad = self.criterion:backward(output, targets[j])
        local rep_grad = self.sim_module:backward(inputs, sim_grad)
      end
      train_loss = train_loss + loss
      loss = loss / batch_size
      -- print(loss)
      self.grad_params:div(batch_size)

      -- regularization
      -- print('loss before:',loss)
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2 * batch_size/dataset.size
      -- print('loss after:',loss)
      self.grad_params:add(self.reg, self.params)
--      count = count + 1
--      print(count)
      return loss, self.grad_params
    end

--    print('Check the gradients:', self.grad_params:size(1)*2)
--    diff, dc, dc_est = optim.checkgrad(feval, self.params:clone())
--    print('Diff must be close to 1e-8: diff = ' .. diff)
    optim.sgd(feval, self.params, self.optim_state)
  end
  xlua.progress(dataset.size, dataset.size)
  train_loss = train_loss/dataset.size
  print('Train loss', train_loss)
end

-- Predict the similarity of a sentence pair.
function AverageVectTrace:predict(lsent, rsent)
  local linputs = self.emb_vecs:index(1, lsent:long())
  local rinputs = self.emb_vecs:index(1, rsent:long())
  local linput_aver = torch.sum(linputs, 1):div(linputs:size(1))
  local rinput_aver = torch.sum(rinputs, 1):div(rinputs:size(1))
  -- get sentence representations
  local inputs = {linput_aver, rinput_aver}

  local output = self.sim_module:forward(inputs)
  return output
end

-- Produce similarity predictions for each sentence pair in the dataset.
function AverageVectTrace:predict_dataset(dataset)
  local predictions = {}
  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    local lsent, rsent = dataset.lsents[i], dataset.rsents[i]
    local output = self:predict(lsent, rsent)
    predictions[i] = torch.exp(output)
  end
  return predictions
end

function AverageVectTrace:compute_loss_dataset(dataset)
  local targets = dataset.labels
  local loss = 0
  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    local lsent, rsent = dataset.lsents[i], dataset.rsents[i]
    local output = self:predict(lsent, rsent)
    local example_loss = self.criterion:forward(output, targets[i])
    loss = loss + example_loss
  end
  loss = loss/dataset.size
  return loss
end

function AverageVectTrace:print_config()
  local num_params = self.params:nElement()
  local num_sim_params = self:new_sim_module():getParameters():nElement()
  printf('%-25s = %d\n',   'num params', num_params)
  printf('%-25s = %d\n',   'word vector dim', self.emb_dim)
  printf('%-25s = %.2e\n', 'regularization strength', self.reg)
  printf('%-25s = %d\n',   'minibatch size', self.batch_size)
  printf('%-25s = %.2e\n', 'learning rate', self.learning_rate)
  printf('%-25s = %d\n',   'sim module hidden dim', self.sim_nhidden)
end

--
-- Serialization
--

function AverageVectTrace:save(path)
  local config = {
    batch_size    = self.batch_size,
    emb_vecs      = self.emb_vecs,
    learning_rate = self.learning_rate,
    hidden_dim     = self.hidden_dim,
    sim_nhidden   = self.sim_nhidden,
    reg           = self.reg,
    structure     = self.structure,
  }

  torch.save(path, {
    params = self.params,
    config = config,
  })
end

function AverageVectTrace.load(path)
  local state = torch.load(path)
  local model = tracenn.AverageVectTrace.new(state.config)
  model.params:copy(state.params)
  return model
end
