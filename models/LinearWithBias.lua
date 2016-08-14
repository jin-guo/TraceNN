local LinearWithBias, parent = torch.class('nn.LinearWithBias', 'nn.Linear')

function LinearWithBias:__init(inputSize, outputSize, biasScala)
    parent.__init(self, inputSize, outputSize)
    self:setBias(biasScala)
end

function LinearWithBias:setBias(biasScala)
    self.bias:fill(biasScala)
end
