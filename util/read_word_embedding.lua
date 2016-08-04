require('init')

opt = {
	binfilename = tracenn.data_dir .. '/wordembedding/word2vec_output/wiki_ptc_nosymbol_100d_w5_i5_vecs.txt',
	outVecs = tracenn.data_dir .. '/wordembedding/wiki_ptc_nosymbol_100d_w5_i5_vecs.vecs',
  outVocab = tracenn.data_dir .. '/wordembedding/wiki_ptc_nosymbol_100d_w5_i5_vecs.vocab'
}

-- Read the trace vocabulary.
local vocab = tracenn.Vocab(tracenn.data_dir .. '/trace_balanced/vocab_ptc_artifact_clean.txt')
Print('Read trace vocabulary with word count:', vocab.size)

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


local emb_vecs = {}
local emb_vocab = {}
--Reading Contents

local i = 0
local contained_word_count = 0
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
		-- Only save the word that is in the trace vocabulary
		if vocab:contains(str) then
			contained_word_count = contained_word_count+1
			emb_vecs[contained_word_count] = vecrep
			emb_vocab[contained_word_count] = str
		end
	end
  i = i + 1
end
collectgarbage()
contained_word_count = contained_word_count-1
print('Total # trace vocabulary found in embedding file:', contained_word_count)


print('Writing Vectors File.')
torch.save(opt.outVecs,emb_vecs)
emb_vecs = nil
collectgarbage()

print('writing Vobab File.')
local vocab_file = torch.DiskFile(opt.outVocab, 'w')
for i = 1, contained_word_count do
  vocab_file:writeString(emb_vocab[i])
  if(i<count) then vocab_file:writeString('\n') end
end
vocab_file:close()
