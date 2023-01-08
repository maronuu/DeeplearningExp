# 4.2.3
for_test.pngに対して正しい出力
# 4.2.5
## unit=3
time: 2m14s
accuracy: 47.96666666666667%
## unit=10000
time: 8m23s
accuracy: 95.11666666666666%

# 4.2.6
CNN
time: 2m24s
accuracy: 96.66666666666667%

# 4.2.8
epoch=7
```
GPU: 0
# Minibatch-size: 100

Accuracy of airplane :  0 %
Accuracy of automobile :  0 %
Accuracy of  bird :  0 %
Accuracy of   cat :  0 %
Accuracy of  deer :  0 %
Accuracy of   dog : 80 %
Accuracy of  frog : 25 %
Accuracy of horse :  0 %
Accuracy of  ship :  0 %
Accuracy of truck :  0 %
Accuracy : 10.500 %
```

## 4.3.1
```bash
$ python search.py -i ./data/mini_mnist/test/4/6.png 
/Users/yutarooguri/dev/deeplearning_exp/.venv/lib/python3.9/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/Users/yutarooguri/dev/deeplearning_exp/.venv/lib/python3.9/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
./data/mini_mnist/train/./9/54.png
./data/mini_mnist/train/./4/139.png
./data/mini_mnist/train/./4/115.png
./data/mini_mnist/train/./5/175.png
./data/mini_mnist/train/./4/131.png
```

# 4.3.2
```bash
$ python search.py -i ./data/mini_mnist/test/4/6.png 
/Users/yutarooguri/dev/deeplearning_exp/.venv/lib/python3.9/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/Users/yutarooguri/dev/deeplearning_exp/.venv/lib/python3.9/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
src_feature.shape : torch.Size([1, 4096])
./data/mini_mnist/train/./9/54.png
./data/mini_mnist/train/./4/139.png
./data/mini_mnist/train/./4/115.png
./data/mini_mnist/train/./5/175.png
./data/mini_mnist/train/./4/131.png
```

# 4.3.3
```python
for i,model in enumerate(self.sequentials):
    x = model(x)
    # Return output of 1st layer in sequentials
    if i == 0:
        return x
```

と変更

同じクエリに対して
```bash
$ python search.py -i ./data/mini_mnist/test/4/6.png   
/Users/yutarooguri/dev/deeplearning_exp/.venv/lib/python3.9/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/Users/yutarooguri/dev/deeplearning_exp/.venv/lib/python3.9/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
src_feature.shape : torch.Size([1, 4096])
./data/mini_mnist/train/./9/54.png
./data/mini_mnist/train/./4/64.png
./data/mini_mnist/train/./4/142.png
./data/mini_mnist/train/./4/115.png
./data/mini_mnist/train/./4/139.png
```
top5の精度は良くなった。

# 4.3.4
MNISTでMLP学習
model weight path: `./mnist_mlp_result`

コードは`search_mymlp.py, create_db_mymlp.py`
```bash
$ python search_mymlp.py -i ./data/mini_mnist/test/4/6.png
src_feature.shape : torch.Size([1, 3, 784, 1000])
data/mini_mnist/train/./7/158.png
data/mini_mnist/train/./4/2.png
data/mini_mnist/train/./4/92.png
data/mini_mnist/train/./6/18.png
data/mini_mnist/train/./4/26.png
```
