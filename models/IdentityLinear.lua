require 'nn'


local IdentityLinear, parent = torch.class('nn.IdentityLinear', 'nn.Linear')

  -- override
  function IdentityLinear:__init(inputSize, outputSize)
      parent.__init(self,inputSize,outputSize)
      if self.inputSize == self.outputSize then
          self.weight = torch.eye(outputSize)
          self.bias:zero()
      end
  end
