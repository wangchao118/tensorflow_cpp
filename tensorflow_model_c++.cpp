#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

using namespace tensorflow;

//加载tensorflow模型
Session *session;
cout << "start initalize session" << "\n";
Status status = NewSession(SessionOptions(), &session);
if (!status.ok()) {
     cout << status.ToString() << "\n";
     return 1;
  }
GraphDef graph_def;
status = ReadBinaryProto(Env::Default(),MNIST_MODEL_PATH, &graph_def);
//MNIST_MODEL_PATH为模型的路径，即model_frozen.pb的路径
if (!status.ok()) {
     cout << status.ToString() << "\n";
     return 1;
  }
status = session->Create(graph_def);
if (!status.ok()) {
     cout << status.ToString() << "\n";
     return 1;
 }
cout << "tensorflow加载成功" << "\n";


Tensor x(DT_FLOAT, TensorShape({1, 784}));//定义输入张量，包括数据类型和大小。
std::vector<float> mydata;　　　　　　　　//输入数据，784维向量
auto dst = x.flat<float>().data();　　　　　
copy_n(mydata.begin(), 784, dst);      //复制mydata到dst
vector<pair<string, Tensor>> inputs = {
    { "input", x}
};                                     //定义模型输入
vector<Tensor> outputs;                //定义模型输出
Status status = session->Run(inputs, {"softmax"}, {}, &outputs);　　//调用模型,
//输出节点名为softmax,结果保存在output中。
if (!status.ok()) {
    cout << status.ToString() << "\n";
    return 1;
}
//get the final label by max probablity
Tensor t = outputs[0];                   // Fetch the first tensor
int ndim = t.shape().dims();             // Get the dimension of the tensor
auto tmap = t.tensor<float, 2>();       
 // Tensor Shape: [batch_size, target_class_num]
 // int output_dim = t.shape().dim_size(1); 
 // Get the target_class_num from 1st dimension
//将结果保存在softmax数组中（该模型是多输出模型）
double softmax[9];  
for (int j = 1; j < 10; j++) {
    softmax[j-1]=tmap(0, j);
}