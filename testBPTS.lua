require 'BPTS'

-- create an arbritrary dictionary
function createDictionary(size)
	local dict = {}
	dict["A"] = torch.rand(size)
	dict["B"] = torch.rand(size)
	dict["C"] = torch.rand(size)
	dict["D"] = torch.rand(size)
	dict["E"] = torch.rand(size)
	dict["F"] = torch.rand(size)
	return dict	
end

function createEncoder(encodingSize)
	local encoder = nn.Sequential()
	encoder:add(nn.Linear(2*encodingSize, encodingSize))
	encoder:add(nn.Tanh())	
	return encoder
end

function main()
	torch.manualSeed(42)
	local encodingSize = 5

	-- parse a manually create tree
	local tree = Tree.parse("(root A (childA (childB B C) D))", createDictionary(encodingSize))

	-- create the encoder to be used at each leaf
	local encoder = createEncoder(encodingSize)

	-- create the network using the tree
	local bpts, input = BPTS.createFromTree(tree, encoder)
	local output = torch.Tensor{1, 0, 0, 0, 0}
	local criterion = nn.MSECriterion()

	-- do a basic functionality test
	criterion:forward(bpts:forward(input), output)
	bpts:zeroGradParameters()
	bpts:backward(input, criterion:backward(bpts.output, output))
	
	-- however, no simple way to test the gradient (below will fail)
	-- local err = Jacobian.testJacobian(bpts, input)
	-- print("error: ", err)	
end


main()