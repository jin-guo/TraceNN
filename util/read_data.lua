--[[

  Functions for loading data from disk.

--]]
package.path = './util/?.lua;' .. package.path

function tracenn.read_embedding(vocab_path, emb_path)
  local vocab = tracenn.Vocab(vocab_path)
  local embedding = torch.load(emb_path)
  return vocab, embedding
end

function tracenn.read_sentences(path, vocab)
  local sentences = {}
  local file = io.open(path, 'r')
  local line
  while true do
    line = file:read()
    if line == nil then break end
    local tokens = stringx.split(line)
    local len = #tokens
    local sent = torch.IntTensor(len)
    for i = 1, len do
      local token = tokens[i]
      sent[i] = vocab:index(token)
    end
    sentences[#sentences + 1] = sent
  end

  file:close()
  return sentences
end


function tracenn.read_trace_dataset(dir, vocab)
  local dataset = {}
  dataset.vocab = vocab
  dataset.lsents = tracenn.read_sentences(dir .. 'a.txt', vocab)
  dataset.rsents = tracenn.read_sentences(dir .. 'b.txt', vocab)
  dataset.size = #dataset.lsents
  local id_file = torch.DiskFile(dir .. 'id.txt')
  local sim_file = torch.DiskFile(dir .. 'sim.txt')
  dataset.ids = torch.IntTensor(dataset.size)
  dataset.labels = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    dataset.ids[i] = id_file:readInt()
  -- Jin: For Tracing, two categories is defined: 2 for link and 1 for no link
    dataset.labels[i] = sim_file:readInt()
  end
  id_file:close()
  sim_file:close()
  return dataset
end
