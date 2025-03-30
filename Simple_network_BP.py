import numpy as np
from drops import calculate

# Sigmoid 激活函数及其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def save_model(filename, model):
    np.savez(filename, 
             weights_input_hidden=model.weights_input_hidden, 
             weights_hidden_output=model.weights_hidden_output, 
             bias_hidden=model.bias_hidden, 
             bias_output=model.bias_output)
    print(f"模型已保存到 {filename}.npz")
def load_model(filename, model):
    data = np.load(filename)
    model.weights_input_hidden = data['weights_input_hidden']
    model.weights_hidden_output = data['weights_hidden_output']
    model.bias_hidden = data['bias_hidden']
    model.bias_output = data['bias_output']
    print(f"模型已加载自 {filename}")

# 误差逆传播神经网络
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # 初始化权重和偏置（使用小随机数）
        self.weights_input_hidden = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))
        self.bias_hidden = np.random.uniform(-1, 1, (1, self.hidden_size))
        self.bias_output = np.random.uniform(-1, 1, (1, self.output_size))

    def forward(self, X):
        # 前向传播
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)
        return self.final_output

    def backward(self, X, y, output):
        # 计算误差
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # 更新权重和偏置
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate

        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if epoch % (epochs // 5) == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        return self.forward(X)


# 读取数据集
X = []
Y = []
with open('Comparison_Binding_energy.txt') as f :
    for lines in f:
        numbers = lines.split()
        x = [int(numbers[0]),int(numbers[1])]
        y = [float(numbers[2])/10000-calculate(x[0],x[1])]  # 从文件中提取A与N，并将实验值与理论值相减
        X.append(x)
        Y.append(y)
# 构建训练集和测试集
Xt = []
Xp = []
Yt = []
Yp = []
# 训练集
xt_start = 20  #训练集开始位置
xt_over = 50 #训练集结束位置（取不到）
xt_step = 2
xp_start = 21 # 预测集开始位置
xp_over = 32 # 预测集结束位置(取不到)
xp_step = 2
for i in range(xt_start,xt_over,xt_step):
    xt = X[i]
    yt = Y[i]
    Xt.append(xt)
    Yt.append(yt)
    
Xt = np.array(Xt) #train

Yt = np.array(Yt) #train

# 创建并训练神经网络
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)
nn.train(Xt, Yt, epochs=10000)

MSE = 0
E_n = 0
# 预测集
for i in range(xp_start,xp_over,xp_step):
    xp = X[i]
    yp = Y[i][0]
    predictions = nn.predict(xp)[0][0]
    MSE += pow((yp-predictions),2)
    E_n += 1
    print(f"Predictions:{predictions},reality:{yp}")
print(f'MSE:{MSE} ')


## 储存模型
# save_model("bp_model.npz", nn)
# current_file = os.path.abspath(__file__)
# current_dir = os.path.dirname(current_file)
# target_file = os.path.join(current_dir, 'bp_model')

## 加载模型
# nn2 = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)
# load_model("bp_model.npz", nn2)