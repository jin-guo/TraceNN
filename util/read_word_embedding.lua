
opt = {
	binfilename = '/Users/Jinguo/Documents/ICSE_tools/tracenn/data/wordembedding/ptc_pagesep_w5_100d_10iter.txt',
	outVecs = '/Users/Jinguo/Documents/ICSE_tools/tracenn/data/wordembedding/ptc_pagesep_w5_100d_10iter.vecs',
  outVocab = '/Users/Jinguo/Documents/ICSE_tools/tracenn/data/wordembedding/ptc_pagesep_w5_100d_10iter.vocab'
}
--Reading the size
local count = 0
local dim = -1
for line in io.lines(opt.binfilename) do
    if count == 1 then
        for i in string.gmatch(line, "%S+") do
            dim = dim + 1
        end
    end
    count = count + 1
end
count = count-1

print("Reading embedding file with ".. count .. ' words of ' .. dim .. ' dimensions.' )

local emb_vecs = torch.DoubleTensor(count,dim)
local emb_vocab = {}
--Reading Contents

local i = 0
for line in io.lines(opt.binfilename) do
	if(i > 0) then
	  xlua.progress(i,count)
	  local vecrep = {}
	  for i in string.gmatch(line, "%S+") do
	    table.insert(vecrep, i)
	  end
	  str = vecrep[1]
	  table.remove(vecrep,1)
		vecrep = torch.DoubleTensor(vecrep)
		local norm = torch.norm(vecrep,2)
		if norm ~= 0 then vecrep:div(norm) end
		emb_vecs[i] = vecrep
	  emb_vocab[i] = str
	end
  i = i + 1
end
collectgarbage()


print('Writing Vectors File.')
torch.save(opt.outVecs,emb_vecs)
emb_vecs = nil
collectgarbage()

print('writing Vobab File.')
local vocab_file = torch.DiskFile(opt.outVocab, 'w')
for i = 1, count do
  vocab_file:writeString(emb_vocab[i])
  if(i<count) then vocab_file:writeString('\n') end
end
vocab_file:close()
