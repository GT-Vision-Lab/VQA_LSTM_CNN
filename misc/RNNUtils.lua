
--------UTIL FUNCTIONS---------------
--Takes a table and return a table which swaps key and value
function reverse(tbl)
	local rev = {}
	print(ipairs(tbl))
	for k, v in ipairs(tbl) do
		rev[v] = k
	end
	return rev
end
--Takes an index (y=x[index]) and return its inverse (x=y[index])
function inverse_mapping(ind)
	local a,b=torch.sort(ind,false)
	return b
end
--Join and split CudaTensors
jtnn=nn.JoinTable(1)
if opt.gpuid >= 0 then
	jtnn = jtnn:cuda()
end
function join_vector(tensor_table)
	return jtnn:forward(tensor_table):clone();
end
function split_vector(w,sizes)
	local tensor_table={};
	local offset=1;
	local n;
	if type(sizes)=="table" then
		n=#sizes;
	else
		n=sizes:size(1);
	end
	for i=1,n do
		table.insert(tensor_table,w[{{offset,offset+sizes[i]-1}}]);
		offset=offset+sizes[i];
	end
	return tensor_table;
end

-------SEQUENCE PROCESSING HELPER FUNCTIONS-----------
function onehot(ind,vocabulary_size)
	local n=ind:size(1);
	local v=torch.FloatTensor(n,vocabulary_size):fill(0);
	v:scatter(2,ind:view(-1,1),1);
	return v;
end
function onehot_cuda(ind,vocabulary_size)
	local n=ind:size(1);
	local v=torch.CudaTensor(n,vocabulary_size):fill(0);
	v:scatter(2,ind:view(-1,1),1);
	return v;
end
function right_align(seq,lengths)
	local v=seq:clone():fill(0);
	local N=seq:size(2);
	for i=1,seq:size(1) do
		v[i][{{N-lengths[i]+1,N}}]=seq[i][{{1,lengths[i]}}];
	end
	return v;
end


-------RNN UTIL FUNCTIONS-----------
--duping an RNN block into multiple ones
function dupe_rnn(net,times)
	local w,dw=net:getParameters();
	local net_arr={};
	local net_dwarr={};
	local net_warr={};
	for i=1,times do
		local tmp=net:clone();
		local tmp1,tmp2=tmp:getParameters();
		--tmp1:set(w); --new test
		table.insert(net_arr,tmp);
		table.insert(net_warr,tmp1);
		table.insert(net_dwarr,tmp2);
	end
	collectgarbage();
	return {net_arr,net_warr,net_dwarr};
end
--sort encoding onehot left aligned
if opt.gpuid >= 0 then
	function sort_encoding_onehot_right_align(batch_word_right_align,batch_length,vocabulary_size)
		--batch_word_right_align: batch_size x MAX_LENGTH matrix, words are right aligned.
		--batch_length: batch_size x 1 matrix depicting actual length
		local batch_length_sorted,sort_index=torch.sort(batch_length,true);
		local sort_index_inverse=inverse_mapping(sort_index);
		local D=batch_word_right_align:size(2);
		local L=batch_length_sorted[1];
		local batch_word_right_align_t=batch_word_right_align:index(1,sort_index):cuda():t()[{{D-L+1,D}}];
		local words=torch.LongTensor(torch.sum(batch_length)):cuda();
		local batch_sizes=torch.LongTensor(L);
		local cnt=0;
		for i=1,L do
			local ind=batch_length_sorted:ge(L-i+1);
			local n=torch.sum(ind);
			words[{{cnt+1,cnt+n}}]=batch_word_right_align_t[i][{{1,n}}];
			batch_sizes[i]=n;
			cnt=cnt+n;
		end
		return {onehot_cuda(words,vocabulary_size),batch_sizes,sort_index,sort_index_inverse};	
	end
else
	function sort_encoding_onehot_right_align(batch_word_right_align,batch_length,vocabulary_size)
		--batch_word_right_align: batch_size x MAX_LENGTH matrix, words are right aligned.
		--batch_length: batch_size x 1 matrix depicting actual length
		local batch_length_sorted,sort_index=torch.sort(batch_length,true);
		local sort_index_inverse=inverse_mapping(sort_index);
		local D=batch_word_right_align:size(2);
		local L=batch_length_sorted[1];
		local batch_word_right_align_t=batch_word_right_align:index(1,sort_index):t()[{{D-L+1,D}}];
		local words=torch.LongTensor(torch.sum(batch_length))
		local batch_sizes=torch.LongTensor(L);
		local cnt=0;
		for i=1,L do
			local ind=batch_length_sorted:ge(L-i+1);
			local n=torch.sum(ind);
			words[{{cnt+1,cnt+n}}]=batch_word_right_align_t[i][{{1,n}}];
			batch_sizes[i]=n;
			cnt=cnt+n;
		end
		return {onehot(words,vocabulary_size),batch_sizes,sort_index,sort_index_inverse};	
	end
end	

--rnn forward, tries to handle most cases
function rnn_forward(net_buffer,init_state,inputs,sizes)
	local N=sizes:size(1);
	local states={init_state[{{1,sizes[1]},{}}]};
	local outputs={};
	for i=1,N do
		local tmp;
		if i==1 or sizes[i]==sizes[i-1] then
			tmp=net_buffer[1][i]:forward({states[i],inputs[i]});
		elseif sizes[i]>sizes[i-1] then
			--right align
			local padding=init_state[{{1,sizes[i]},{}}];
			padding[{{1,sizes[i-1]},{}}]=states[i];
			states[i]=padding;
			tmp=net_buffer[1][i]:forward({padding,inputs[i]});
		elseif sizes[i]<sizes[i-1] then
			--left align
			tmp=net_buffer[1][i]:forward({states[i][{{1,sizes[i]}}],inputs[i]});
		end
		table.insert(states,tmp[1]);
		table.insert(outputs,tmp[2]);
	end
	return states,outputs;
end
--rnn backward
function rnn_backward(net_buffer,dend_state,doutputs,states,inputs,sizes)
	if type(doutputs)=="table" then
		local N=sizes:size(1);
		local dstate={[N+1]=dend_state[{{1,sizes[N]},{}}]};
		local dinput_embedding={};
		for i=N,1,-1 do
			local tmp;
			if i==1 or sizes[i]==sizes[i-1] then
				tmp=net_buffer[1][i]:backward({states[i],inputs[i]},{dstate[i+1],doutputs[i]});
				dstate[i]=tmp[1];
			elseif sizes[i]>sizes[i-1] then
				--right align
				tmp=net_buffer[1][i]:backward({states[i],inputs[i]},{dstate[i+1],doutputs[i]});
				dstate[i]=tmp[1][{{1,sizes[i-1]},{}}];
			elseif sizes[i]<sizes[i-1] then
				--left align
				--compute a larger dstate that matches i-1
				tmp=net_buffer[1][i]:backward({states[i][{{1,sizes[i]}}],inputs[i]},{dstate[i+1],doutputs[i]});
				local padding=dend_state[{{1,sizes[i-1]},{}}];
				padding[{{1,sizes[i]},{}}]=tmp[1];
				dstate[i]=padding;
			end
			dinput_embedding[i]=tmp[2];
		end
		return dstate,dinput_embedding;
	else
		local N=sizes:size(1);
		local dstate={[N+1]=dend_state[{{1,sizes[N]},{}}]};
		local dinput_embedding={};
		for i=N,1,-1 do
			local tmp;
			if i==1 or sizes[i]==sizes[i-1] then
				tmp=net_buffer[1][i]:backward({states[i],inputs[i]},{dstate[i+1],torch.repeatTensor(doutputs,sizes[i],1)});
				dstate[i]=tmp[1];
			elseif sizes[i]>sizes[i-1] then
				--right align
				tmp=net_buffer[1][i]:backward({states[i],inputs[i]},{dstate[i+1],torch.repeatTensor(doutputs,sizes[i],1)});
				dstate[i]=tmp[1][{{1,sizes[i-1]},{}}];
			elseif sizes[i]<sizes[i-1] then
				--left align
				--compute a larger dstate that matches i-1
				tmp=net_buffer[1][i]:backward({states[i][{{1,sizes[i]}}],inputs[i]},{dstate[i+1],torch.repeatTensor(doutputs,sizes[i],1)});
				local padding=dend_state[{{1,sizes[i-1]},{}}];
				padding[{{1,sizes[i]},{}}]=tmp[1];
				dstate[i]=padding;
			end
			dinput_embedding[i]=tmp[2];
		end
		return dstate,dinput_embedding;
	end	
end