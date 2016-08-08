--[[

  Functions for loading data from disk.

--]]
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

function tracenn.read_artifact(dir, vocab)
  local artifact = {}
  artifact.vocab = vocab
  artifact.src_artfs = tracenn.read_sentences(dir .. 'src_artf/sentence.txt', vocab)
  artifact.trg_artfs = tracenn.read_sentences(dir .. 'trg_artf/sentence.txt', vocab)
  local src_id_file = torch.DiskFile(dir .. 'src_artf/id.txt')
  local trg_id_file = torch.DiskFile(dir .. 'trg_artf/id.txt')
  artifact.src_artfs_ids = {}
  artifact.trg_artfs_ids = {}
  for i = 1, #artifact.src_artfs do
    artifact.src_artfs_ids[src_id_file:readString("*l")] = i
  end
  for i = 1, #artifact.trg_artfs do
    artifact.trg_artfs_ids[trg_id_file:readString("*l")] = i
  end
  src_id_file:close()
  trg_id_file:close()
  return artifact
end


function tracenn.read_trace_dataset(dir, vocab)
  local dataset = {}
  dataset.vocab = vocab
  dataset.lsents = {}
  local a_file = io.open(dir .. 'a.txt')
  if a_file then
    for line in a_file:lines() do
        dataset.lsents[#dataset.lsents + 1] = line
    end
  end

  dataset.rsents = {}
  local b_file = io.open(dir .. 'b.txt')
  if b_file then
    for line in b_file:lines() do
      dataset.rsents[#dataset.rsents + 1] = line
    end
  end

  a_file.close()
  b_file.close()

  dataset.size = #dataset.lsents
  local id_file = torch.DiskFile(dir .. 'id.txt')
  local sim_file = torch.DiskFile(dir .. 'sim.txt')
  dataset.ids = {}
  dataset.labels = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    dataset.ids[i] = id_file:readString("*l")
  -- Jin: For Tracing, two categories is defined: 2 for link and 1 for no link
    dataset.labels[i] = sim_file:readInt()
  end
  id_file:close()
  sim_file:close()
  return dataset
end
