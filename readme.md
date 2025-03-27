### 误差逆传播算法实现预测原子核结合能

可在文件中修改xt_start与xt_over，xt_step决定训练集开始位置，结束位置以及步长，暂时仅支持等差数列。
也可修改xp_start，xp_over，xp_step决定预测集开始位置，结束位置以及步长，同样仅支持等差数列

训练轮次可修改epochs值，默认10000

储存模型文件和加载模型文件在文件最下面被我注释掉了，如需使用请将注释删去

一般来说都有numpy库吧但为了以防万一还是加了requirements.txt

请在本项目文件夹下运行cmd，执行代码`python -m pip install -r requirements.txt`或直接`pip install numpy`

~~bp算法由ai生成~~
