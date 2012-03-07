--[[--
In backpropagation through structure (BPTS) [see: "Learning task-dependent distributed representations by backpropagation 
through structure" (Goller and Kuchler, 1996)] a tree is created from the input, with a single encoder is cloned per node. 
Forward propagation is carried out by taking the input of the leafs and concatenating them in the parent
node (similar to RAAM). Thus for the tree:

		  A
		/	\
	   B     C
	   
A's input is the output of B and C concatenated together. The error is calculated for the top node (given
a criterion), with backpropagation carried out by splitting the error across the children.
--]]--

local BPTS = torch.class("BPTS")

require 'nn'
require 'Tree'


function BPTS.createNode(encoder, children)
	local seq = nn.Sequential()
	
	-- set children as input to this node
	local parallel = nn.ParallelTable()	
	for _, child in ipairs(children) do			
		parallel:add(child)
	end
	seq:add(parallel)
		
	-- concatenate their values
	seq:add(nn.JoinTable(1))		
	
	local sharedEncoder = encoder:clone("weight", "bias")
	seq:add(sharedEncoder)
	return seq
end

function BPTS.createFromTree(tree, encoder)
	if tree:isLeaf() then
		-- leafs' inputs are their values
		return nn.Identity(), tree.value
	else
		local children = {}
		local values = {}
		for _, child in ipairs(tree.children) do
			mod, value = BPTS.createFromTree(child, encoder)			
			table.insert(children, mod)
			table.insert(values, value)
		end
		
		return BPTS.createNode(encoder, children), values
	end
end
