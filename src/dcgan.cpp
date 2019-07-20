#include <torch/torch.h>
using namespace std;

class Net : public torch::nn::Module {
public:
	Net(initializer_list<int> li) {
		int prev = -1;
		int i = 0;
		for(int k : li) {
			if(prev >= 0) 
				v.emplace_back(register_module("linear" + to_string(++i),
							torch::nn::Linear{prev, k}));
			prev = k;
		}
	}
	torch::Tensor forward(torch::Tensor input) {
		for(auto &a : v) input = torch::relu(a->forward(input));
		return input;
	}

	vector<torch::nn::Linear> v;
};

int main() {
	Net net{4, 5, 5, 2};
	for (const auto& pair : net.named_parameters()) {
		cout << pair.key() << " :\n" << pair.value() << "\n\n";
	}

	torch::Tensor in = torch::rand({1,4});
	auto out = net.forward(in);
	//torch::Tensor target = torch::rand({1,2});
	torch::optim::SGD optimizer{net.parameters(), 0.01};
	optimizer.zero_grad();
	auto loss = torch::nll_loss(out, out);
	loss.backward();
	optimizer.step();
//	cout << net.v[0]->grad() << endl;
//	cout << in.grad() << endl << out << endl;
	torch::serialize::OutputArchive output_archive;
	net.save(output_archive);
	output_archive.save_to("net2.pt");
}

