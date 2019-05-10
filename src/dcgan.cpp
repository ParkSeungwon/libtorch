#include <torch/torch.h>
using namespace std;

class Net : public torch::nn::Module {
public:
	Net(int64_t N, int64_t M)
		: W{register_module("Linear", torch::nn::Linear(N, M))}
		, b{register_parameter("bias", torch::randn(M))}
	{ }
	torch::Tensor forward(torch::Tensor input) {
		return W(input) + b;
	}

protected:
	torch::nn::Linear W;
	torch::Tensor b;
};

int main() {
	Net net(4, 5);
	for (const auto& pair : net.named_parameters()) {
		cout << pair.key() << " :\n" << pair.value() << "\n\n";
	}

	auto tensor = torch::eye(3);
	cout << tensor << endl;
	cout << net.forward(torch::ones({2,4})) << endl;
}

