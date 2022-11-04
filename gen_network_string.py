def gen_string(network_layers):
    """Generates code for network configuration. For example when given input of '10-20' it will output 'self.linear_regression = nn.Sequential(nn.Linear(input_size,10),nn.Tanh(),nn.Linear(10,20),nn.ReLU(),nn.Linear(20,output))'

    Args:
        network_layers (string): string in the form '10-5-10' where 10,5,10 are hidden layers

    Returns:
        code for neural network configuration as already mentioned
    """
    nums = [i for i in network_layers.split("-")]
    first_line = f"nn.Linear(input_size,{nums[0]}),"
    second_last_line = "nn.ReLU(),"
    last_line = f"nn.Linear({nums[-1]},output_size)"

    intermediate_lines = ""

    begin = "self.linear_regression = nn.Sequential("
    end = ")"
    if len(nums)>1:
        for i in range(len(nums)-1):
            first,second = nums[i],nums[i+1]
            current_line = f"nn.Linear({first},{second}),"
            intermediate_lines += "nn.Tanh()," + current_line

    inside =  first_line+intermediate_lines+second_last_line+last_line
    return begin+inside+end

if __name__ == '__main__':
    x = gen_string("10")
