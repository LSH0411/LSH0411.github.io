{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "a3bb7dfd",
   "metadata": {},
   "outputs": [],
   "source": [
    "## 선형회귀\n",
    "import torch ## what? pytorch 라이브러리 임포트\n",
    "import torch.nn as nn ## pytorch 라이브러리에 있는 신경망 구성요소 import nn = neural network\n",
    "import torch.optim as optim ## pytorch 라이브러리에 있는 최적화 함수들 import, optim = optimizer\n",
    "import torch.nn.functional as F ## 신경망 관련 함수(loss) import"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "94166a5b",
   "metadata": {},
   "outputs": [],
   "source": [
    "## 데이터 -> pytorch는 tensor 사용\n",
    "## 선형적인 데이터 선언 -> (1, 2), (2, 4), (3, 6) 그렇다면 인풋과 아웃풋이 어떻게 돼야하나? \n",
    "## 인풋의 예시는 1이 들어오면 아웃풋이 2가 나와야 한다.\n",
    "## 모델의 가중치의 형식은 어떻게 돼야 할까? 입력이 3x1이고 아웃풋도 3x1이니 가중치는 1x1, 마찬가지로 바이어스도 1x1이어야 한다. 그러면 출력이 3x1이 될 것이다.\n",
    "x_train = torch.FloatTensor([[1], [2], [3]]) \n",
    "y_train = torch.FloatTensor([[2], [4], [6]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "8009088b",
   "metadata": {},
   "outputs": [],
   "source": [
    "## 모델 만들기\n",
    "class LinearRegression(nn.Module):\n",
    "    def __init__(self):\n",
    "        super(LinearRegression, self).__init__() ## 왜 super 형식이 이렇게 되지? 무슨 일이 일어나는 거지?\n",
    "        self.linear = nn.Linear(1, 1) ## 왜 인자가 1,1 인가? \n",
    "        \n",
    "    def forward(self ,x):\n",
    "        return self.linear(x)\n",
    "    \n",
    "\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "859d44ff",
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "super() argument 1 must be type, not LinearRegression",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[16], line 5\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[39m# 에포크, 손실함수, 최적화 함수 설정\u001b[39;00m\n\u001b[1;32m      3\u001b[0m total_epoch \u001b[39m=\u001b[39m \u001b[39m2000\u001b[39m\n\u001b[0;32m----> 5\u001b[0m model \u001b[39m=\u001b[39m LinearRegression()\n\u001b[1;32m      7\u001b[0m optimizer \u001b[39m=\u001b[39m optim\u001b[39m.\u001b[39mSGD(model\u001b[39m.\u001b[39mparameters(), lr\u001b[39m=\u001b[39m\u001b[39m1e-2\u001b[39m)\n",
      "Cell \u001b[0;32mIn[15], line 4\u001b[0m, in \u001b[0;36mLinearRegression.__init__\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[39mdef\u001b[39;00m \u001b[39m__init__\u001b[39m(\u001b[39mself\u001b[39m):\n\u001b[0;32m----> 4\u001b[0m     \u001b[39msuper\u001b[39;49m(\u001b[39mself\u001b[39;49m, LinearRegression)\u001b[39m.\u001b[39m\u001b[39m__init__\u001b[39m() \u001b[39m## 왜 super 형식이 이렇게 되지? 무슨 일이 일어나는 거지?\u001b[39;00m\n\u001b[1;32m      5\u001b[0m     \u001b[39mself\u001b[39m\u001b[39m.\u001b[39mlinear \u001b[39m=\u001b[39m nn\u001b[39m.\u001b[39mLinear(\u001b[39m1\u001b[39m, \u001b[39m1\u001b[39m)\n",
      "\u001b[0;31mTypeError\u001b[0m: super() argument 1 must be type, not LinearRegression"
     ]
    }
   ],
   "source": [
    "# 에포크, 손실함수, 최적화 함수 설정\n",
    "\n",
    "total_epoch = 2000\n",
    "\n",
    "model = LinearRegression()\n",
    "\n",
    "optimizer = optim.SGD(model.parameters(), lr=1e-2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "c2fd9e43",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0  Loss: 3.185126121185711e-11\n",
      "Epoch: 100  Loss: 2.4101609596982598e-11\n",
      "Epoch: 200  Loss: 2.1088908397359774e-11\n",
      "Epoch: 300  Loss: 1.7678303265711293e-11\n",
      "Epoch: 400  Loss: 1.6825651982799172e-11\n",
      "Epoch: 500  Loss: 1.3718211462621088e-11\n",
      "Epoch: 600  Loss: 1.2979247017430584e-11\n",
      "Epoch: 700  Loss: 1.2979247017430584e-11\n",
      "Epoch: 800  Loss: 1.2979247017430584e-11\n",
      "Epoch: 900  Loss: 1.2278178473934531e-11\n",
      "Epoch: 1000  Loss: 1.2278178473934531e-11\n",
      "Epoch: 1100  Loss: 1.2278178473934531e-11\n",
      "Epoch: 1200  Loss: 1.2278178473934531e-11\n",
      "Epoch: 1300  Loss: 1.2278178473934531e-11\n",
      "Epoch: 1400  Loss: 1.2278178473934531e-11\n",
      "Epoch: 1500  Loss: 1.2278178473934531e-11\n",
      "Epoch: 1600  Loss: 1.2278178473934531e-11\n",
      "Epoch: 1700  Loss: 1.2278178473934531e-11\n",
      "Epoch: 1800  Loss: 1.2278178473934531e-11\n",
      "Epoch: 1900  Loss: 1.2278178473934531e-11\n",
      "Epoch: 2000  Loss: 1.2278178473934531e-11\n"
     ]
    }
   ],
   "source": [
    "for epoch in range(total_epoch+1):\n",
    "    output = model(x_train)\n",
    "    \n",
    "    cost = F.mse_loss(output, y_train)\n",
    "    \n",
    "    optimizer.zero_grad()\n",
    "    \n",
    "    cost.backward()\n",
    "    \n",
    "    optimizer.step()\n",
    "    \n",
    "    if epoch % 100 == 0:\n",
    "        print(\"Epoch: {}  Loss: {}\".format(epoch, cost.item()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "aba1fcaf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "torch.Size([128, 30])\n",
      "Parameter containing:\n",
      "tensor([[-0.1286,  0.0123, -0.0336,  0.1260, -0.0311, -0.1110, -0.1032,  0.2138,\n",
      "         -0.1472,  0.0565, -0.1051, -0.0808, -0.2172,  0.1564,  0.2032, -0.1050,\n",
      "         -0.0203,  0.1608,  0.0788,  0.0643],\n",
      "        [-0.1803,  0.1921,  0.0473,  0.1535,  0.1238, -0.1081, -0.0996,  0.2207,\n",
      "         -0.0972, -0.0353, -0.1556, -0.1671,  0.0912,  0.0817,  0.0977, -0.1423,\n",
      "         -0.0986,  0.0885, -0.1508,  0.0549],\n",
      "        [-0.1791,  0.0227,  0.2174, -0.0562, -0.0714, -0.1652, -0.0226,  0.0383,\n",
      "         -0.1805, -0.1288, -0.1676,  0.0570, -0.0422, -0.1121, -0.2057,  0.1444,\n",
      "         -0.0719,  0.1077,  0.1140,  0.0406],\n",
      "        [ 0.1623, -0.0430, -0.0770, -0.0043,  0.1617,  0.0716, -0.1241,  0.0130,\n",
      "         -0.1434,  0.0909, -0.2226, -0.0710,  0.0712, -0.0824,  0.1602, -0.1986,\n",
      "          0.1067,  0.1848, -0.0823,  0.1360],\n",
      "        [ 0.0152,  0.0118,  0.1434,  0.0448, -0.1405, -0.1253, -0.0076, -0.1100,\n",
      "          0.0150,  0.1687, -0.1166, -0.1302,  0.1132, -0.0767, -0.0651, -0.0339,\n",
      "          0.1044,  0.0496, -0.1898, -0.0603],\n",
      "        [-0.0112, -0.0165, -0.1333, -0.1632,  0.1495, -0.0963, -0.0893,  0.0157,\n",
      "         -0.1743, -0.1140, -0.1907, -0.1353,  0.1353, -0.1376,  0.0014, -0.0158,\n",
      "          0.2213, -0.0443,  0.1443,  0.0636],\n",
      "        [ 0.0626, -0.1461, -0.1871,  0.0494,  0.1952,  0.0903,  0.1143, -0.0405,\n",
      "         -0.1495,  0.1921,  0.0940,  0.1626, -0.0024, -0.1401, -0.1220, -0.0063,\n",
      "         -0.2201, -0.0285, -0.1396, -0.1506],\n",
      "        [ 0.1479, -0.1867,  0.1456, -0.1724, -0.0491, -0.1905, -0.1046, -0.2105,\n",
      "          0.1491,  0.1901, -0.0647,  0.0983, -0.1091, -0.0371, -0.0831, -0.0796,\n",
      "         -0.1755,  0.1268, -0.2156,  0.0689],\n",
      "        [ 0.0304,  0.1754, -0.0701, -0.0103, -0.1402,  0.2125, -0.2142, -0.2208,\n",
      "          0.0544, -0.1829, -0.0686,  0.1038, -0.2152,  0.0583, -0.0442,  0.0822,\n",
      "          0.0594, -0.0995, -0.0186, -0.1175],\n",
      "        [ 0.0833, -0.1850, -0.0800, -0.1515, -0.1350,  0.1739, -0.1558, -0.1160,\n",
      "          0.0153, -0.1062,  0.0112, -0.1300,  0.1752,  0.0399,  0.0739,  0.0042,\n",
      "          0.0591,  0.0556, -0.0737, -0.1650],\n",
      "        [ 0.1547,  0.1494, -0.0735, -0.1686, -0.0967,  0.1969, -0.0475, -0.1165,\n",
      "         -0.1122, -0.1278,  0.0234, -0.1395,  0.0668,  0.0737,  0.0233, -0.1917,\n",
      "          0.0067,  0.1458,  0.0940,  0.0661],\n",
      "        [ 0.0420, -0.2065,  0.0651, -0.1687, -0.2122, -0.0776,  0.1630,  0.1276,\n",
      "         -0.0146,  0.1651,  0.1131,  0.1064, -0.0174,  0.0728,  0.1080,  0.0279,\n",
      "         -0.1380, -0.0018, -0.0039, -0.1629],\n",
      "        [-0.0186,  0.2031,  0.0883, -0.2189, -0.1543, -0.1231,  0.0451, -0.0717,\n",
      "         -0.0165,  0.0267, -0.0010,  0.2014, -0.1183,  0.1398,  0.0633,  0.0778,\n",
      "          0.0355, -0.2046, -0.1395,  0.1612],\n",
      "        [-0.0930, -0.0051, -0.0689, -0.0932,  0.1416,  0.0748,  0.0087, -0.0989,\n",
      "         -0.0969,  0.1355, -0.2162, -0.2047,  0.1901, -0.0451, -0.1876,  0.1836,\n",
      "          0.2132, -0.1178,  0.1871, -0.0092],\n",
      "        [-0.1892,  0.2061,  0.0327,  0.0497, -0.2067,  0.1962,  0.2193,  0.0779,\n",
      "         -0.1118,  0.0146,  0.1808,  0.1836, -0.0773, -0.1276, -0.1272,  0.2200,\n",
      "         -0.0258, -0.0134, -0.1075,  0.0258],\n",
      "        [ 0.0273, -0.1877,  0.0626,  0.1052, -0.1816,  0.2166, -0.0874, -0.0416,\n",
      "          0.1471,  0.1283, -0.0315, -0.1519, -0.0275,  0.2194, -0.0530,  0.0624,\n",
      "          0.0812, -0.2096, -0.1655,  0.0058],\n",
      "        [ 0.1354, -0.1895, -0.1204,  0.1526,  0.1445,  0.1946,  0.0956,  0.2068,\n",
      "          0.1899, -0.1670, -0.1111,  0.2172, -0.1177,  0.1836, -0.1771, -0.2112,\n",
      "          0.2205, -0.1901,  0.1704,  0.1882],\n",
      "        [-0.1941,  0.1620, -0.2143, -0.0927,  0.2077, -0.0669,  0.1710,  0.1843,\n",
      "         -0.1683, -0.0773, -0.0144,  0.1666, -0.2072, -0.1325,  0.1772,  0.1134,\n",
      "          0.0170, -0.0176,  0.0564, -0.1978],\n",
      "        [ 0.1228,  0.1867, -0.1186, -0.1075,  0.1577,  0.2063,  0.1733, -0.0278,\n",
      "          0.1426,  0.0901,  0.2073, -0.1142,  0.0870, -0.0516,  0.1093, -0.0025,\n",
      "         -0.0693, -0.0372,  0.0037, -0.0947],\n",
      "        [ 0.0903,  0.1914, -0.0762, -0.0947, -0.2051, -0.1796,  0.0361,  0.1001,\n",
      "         -0.1817, -0.0141, -0.2027,  0.0074, -0.1163, -0.1402,  0.0029, -0.2104,\n",
      "          0.0314,  0.1531,  0.1494,  0.0116],\n",
      "        [ 0.0697, -0.1017,  0.1143,  0.1992,  0.0921, -0.0941,  0.0041,  0.2070,\n",
      "          0.1041,  0.1163, -0.0251, -0.0318, -0.0695,  0.1350,  0.1607,  0.1805,\n",
      "         -0.0544,  0.1652,  0.2080,  0.1686],\n",
      "        [-0.1449,  0.1576, -0.1582, -0.0437,  0.1924,  0.0585, -0.1235,  0.0998,\n",
      "          0.0480,  0.0885,  0.0387, -0.0728, -0.1697,  0.0521, -0.1672, -0.0555,\n",
      "          0.2203, -0.1448, -0.1231, -0.1358],\n",
      "        [ 0.2060,  0.0188, -0.1296, -0.0652,  0.0971, -0.0750,  0.1571,  0.0508,\n",
      "         -0.2029,  0.0890,  0.1593, -0.0804,  0.0713,  0.2101,  0.2169,  0.1356,\n",
      "          0.2158,  0.0264, -0.1487,  0.1655],\n",
      "        [ 0.0215, -0.1973,  0.1394,  0.0818, -0.1382,  0.0993,  0.2133, -0.1493,\n",
      "          0.0730, -0.0796, -0.0583, -0.1151,  0.1414, -0.1509,  0.2142,  0.0970,\n",
      "         -0.0302,  0.0438, -0.0628,  0.1611],\n",
      "        [-0.0697, -0.0218, -0.1141, -0.0835, -0.1821,  0.2001, -0.2209, -0.0361,\n",
      "         -0.1119,  0.0861, -0.1288,  0.0143,  0.0230,  0.1686, -0.1939, -0.0890,\n",
      "         -0.0918,  0.1710,  0.0641,  0.0779],\n",
      "        [ 0.1408,  0.0395, -0.0321, -0.2112, -0.1913,  0.0993,  0.1672, -0.1512,\n",
      "         -0.0727,  0.0119,  0.0104,  0.1569, -0.1865, -0.0198, -0.1835, -0.0121,\n",
      "         -0.1572,  0.1964,  0.0191,  0.0406],\n",
      "        [ 0.0592,  0.0636, -0.0860,  0.0541,  0.1700, -0.0926,  0.0073,  0.0925,\n",
      "          0.2115, -0.2226,  0.2002,  0.0062, -0.0796,  0.0758,  0.2188, -0.1724,\n",
      "         -0.1910,  0.1470, -0.1768,  0.1863],\n",
      "        [ 0.2145, -0.1986, -0.1178,  0.2010,  0.0026, -0.0053, -0.0627, -0.1482,\n",
      "          0.0597, -0.1787,  0.0849, -0.0972, -0.0013, -0.0683, -0.1286, -0.0419,\n",
      "         -0.0269,  0.1616,  0.1898, -0.0725],\n",
      "        [-0.0121, -0.0824,  0.1113,  0.0179,  0.1954, -0.0093, -0.1838, -0.1681,\n",
      "          0.1830,  0.0839,  0.1464, -0.2115, -0.2104, -0.2079, -0.1210,  0.0806,\n",
      "         -0.2165, -0.1631, -0.2159,  0.0415],\n",
      "        [ 0.2079,  0.0236,  0.0196,  0.1567,  0.1349, -0.0152,  0.1554, -0.1030,\n",
      "          0.0699,  0.0339, -0.2163,  0.1779, -0.0409, -0.1906, -0.0877,  0.0025,\n",
      "          0.0974, -0.1179,  0.0420,  0.1072]], requires_grad=True)\n",
      "Parameter containing:\n",
      "tensor([ 0.0763, -0.1518, -0.0453,  0.1429, -0.1293,  0.1385, -0.0196,  0.1461,\n",
      "         0.0377, -0.2153, -0.0186,  0.0638, -0.1868,  0.2024,  0.1745,  0.1164,\n",
      "        -0.2175, -0.2148, -0.1018, -0.1542, -0.1705,  0.0637,  0.1664,  0.0809,\n",
      "         0.0641, -0.1925,  0.0011, -0.1232,  0.0732, -0.1951],\n",
      "       requires_grad=True)\n"
     ]
    }
   ],
   "source": [
    "m = nn.Linear(20, 30) \n",
    "input = torch.randn(128, 20) \n",
    "output = m(input) \n",
    "print(output.size())\n",
    "print(m.weight)\n",
    "print(m.bias)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "cb22dbb4",
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "forward() missing 1 required positional argument: 'x'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[20], line 2\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[39minput\u001b[39m \u001b[39m=\u001b[39m torch\u001b[39m.\u001b[39mFloatTensor([\u001b[39m7\u001b[39m])\n\u001b[0;32m----> 2\u001b[0m output \u001b[39m=\u001b[39m model\u001b[39m.\u001b[39;49mforward(\u001b[39minput\u001b[39;49m)\n\u001b[1;32m      3\u001b[0m \u001b[39mprint\u001b[39m(output)\n",
      "\u001b[0;31mTypeError\u001b[0m: forward() missing 1 required positional argument: 'x'"
     ]
    }
   ],
   "source": [
    "input = torch.FloatTensor([7])\n",
    "output = model.forward(input)\n",
    "print(output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "aa98a089",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5\n",
      "Alice\n"
     ]
    }
   ],
   "source": [
    "## super\n",
    "\n",
    "class Parent: ## 클래스 선언\n",
    "    def __init__(self, name): ## 생성자, 객체 생성 시 한번 호출, 이때 name을 인자로 받음\n",
    "        self.name = name\n",
    "    \n",
    "    \n",
    "class Child(Parent): ## Child 클래스로 Parent 상속 받음\n",
    "    def __init__(self, name, age): ## Child 객체 생성시 생성자 호출, name과 age를 인자로 받음 \n",
    "        super().__init__(name) ## Parent 생성자 호출\n",
    "        self.age = age\n",
    "        \n",
    "\n",
    "child = Child(\"Alice\", 5)\n",
    "\n",
    "print(child.age)\n",
    "\n",
    "print(child.name)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "f68c05cf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "24\n",
      "4.5\n",
      "lsh\n"
     ]
    }
   ],
   "source": [
    "class Person:\n",
    "    def __init__(self, name, age):\n",
    "        self.name = name\n",
    "        self.age = age\n",
    "\n",
    "    def get_name(self):\n",
    "        print(f'제 이름은 {self.name}입니다.')\n",
    "    \n",
    "    def get_age(self):\n",
    "        print(f'제 나이는 {self.age}세 입니다.')\n",
    "        \n",
    "        \n",
    "class Student(Person):\n",
    "    def __init__(self, name, age, GPA):\n",
    "        super().__init__(name, age)\n",
    "        self.GPA = GPA\n",
    "\n",
    "    def get_GPA(self):\n",
    "        print(f'제 학점은 {self.GPA}입니다.')\n",
    "        \n",
    "s1 = Student(\"lsh\", 24, 4.5)\n",
    "\n",
    "print(s1.age)\n",
    "print(s1.GPA)\n",
    "print(s1.name)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "61d93453",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [1/10], Loss: 0.3752\n",
      "Epoch [2/10], Loss: 0.1055\n",
      "Epoch [3/10], Loss: 0.0345\n",
      "Epoch [4/10], Loss: 0.1355\n",
      "Epoch [5/10], Loss: 0.0701\n",
      "Epoch [6/10], Loss: 0.0227\n",
      "Epoch [7/10], Loss: 0.0108\n",
      "Epoch [8/10], Loss: 0.0040\n",
      "Epoch [9/10], Loss: 0.0441\n",
      "Epoch [10/10], Loss: 0.0024\n",
      "Accuracy: 99.55\n",
      "Accuracy: 97.59\n"
     ]
    }
   ],
   "source": [
    "# 다층 퍼셉트론으로 MNIST 분류\n",
    "\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "import torchvision.datasets as datasets\n",
    "import torchvision.transforms as transforms\n",
    "from torch.utils.data import DataLoader\n",
    "\n",
    "# GPU를 사용할 수 있다면 사용합니다.\n",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "\n",
    "# 하이퍼파라미터 정의\n",
    "input_size = 784  # 28x28 픽셀\n",
    "hidden_size = 128\n",
    "num_classes = 10\n",
    "learning_rate = 0.001\n",
    "batch_size = 64\n",
    "num_epochs = 10\n",
    "\n",
    "# MNIST 데이터셋 로드\n",
    "train_dataset = datasets.MNIST(root='data/',\n",
    "                               train=True,\n",
    "                               transform=transforms.ToTensor(),\n",
    "                               download=True)\n",
    "\n",
    "test_dataset = datasets.MNIST(root='data/',\n",
    "                              train=False,\n",
    "                              transform=transforms.ToTensor())\n",
    "\n",
    "# 데이터 로더 생성\n",
    "train_loader = DataLoader(dataset=train_dataset,\n",
    "                          batch_size=batch_size,\n",
    "                          shuffle=True)\n",
    "\n",
    "test_loader = DataLoader(dataset=test_dataset,\n",
    "                         batch_size=batch_size,\n",
    "                         shuffle=False)\n",
    "\n",
    "# 다층 퍼셉트론 모델 정의\n",
    "class MLP(nn.Module):\n",
    "    def __init__(self, input_size, hidden_size, num_classes):\n",
    "        super(MLP, self).__init__()\n",
    "        self.fc1 = nn.Linear(input_size, hidden_size)\n",
    "        self.relu = nn.ReLU()\n",
    "        self.fc2 = nn.Linear(hidden_size, num_classes)\n",
    "    \n",
    "    def forward(self, x):\n",
    "        out = self.fc1(x)\n",
    "        out = self.relu(out)\n",
    "        out = self.fc2(out)\n",
    "        return out\n",
    "\n",
    "# 모델 생성 및 손실함수, 최적화 함수 정의\n",
    "model = MLP(input_size, hidden_size, num_classes).to(device)\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "optimizer = optim.Adam(model.parameters(), lr=learning_rate)\n",
    "\n",
    "# 학습 루프\n",
    "for epoch in range(num_epochs):\n",
    "    for batch_idx, (data, targets) in enumerate(train_loader):\n",
    "        data = data.to(device=device)\n",
    "        targets = targets.to(device=device)\n",
    "        \n",
    "        # 순전파\n",
    "        scores = model(data.reshape(-1, 28*28))\n",
    "        loss = criterion(scores, targets)\n",
    "        \n",
    "        # 역전파 및 가중치 업데이트\n",
    "        optimizer.zero_grad()\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        \n",
    "    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')\n",
    "\n",
    "# 테스트 루프\n",
    "def test(loader, model):\n",
    "    num_correct = 0\n",
    "    num_samples = 0\n",
    "    model.eval()\n",
    "    \n",
    "    with torch.no_grad():\n",
    "        for data, targets in loader:\n",
    "            data = data.to(device=device)\n",
    "            targets = targets.to(device=device)\n",
    "            \n",
    "            # 순전파\n",
    "            scores = model(data.reshape(-1, 28*28))\n",
    "            _, predictions = scores.max(1)\n",
    "            \n",
    "            # 정확도 계산\n",
    "            num_correct += (predictions == targets).sum()\n",
    "            num_samples += predictions.size(0)\n",
    "    \n",
    "    accuracy = float(num_correct) / float(num_samples) * 100\n",
    "    print(f'Accuracy: {accuracy:.2f}')\n",
    "\n",
    "test(train_loader, model)\n",
    "test(test_loader, model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "4f935d39",
   "metadata": {},
   "outputs": [],
   "source": [
    "## CNN 으로 분류\n",
    "import torch\n",
    "import torch.nn as nn \n",
    "import torch.nn.functional as F\n",
    "import torch.optim as optim \n",
    "from torchvision import datasets, transforms \n",
    "\n",
    "from matplotlib import pyplot as plt\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "c8b1d9af",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Current cuda device is cuda\n"
     ]
    }
   ],
   "source": [
    "is_cuda = torch.cuda.is_available()\n",
    "device = torch.device('cuda' if is_cuda else 'cpu')\n",
    "\n",
    "print('Current cuda device is', device)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "1cdd42d3",
   "metadata": {},
   "outputs": [],
   "source": [
    "batch_size = 50\n",
    "learning_rate = 1e-4\n",
    "epoch_num = 15"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "b3d35a13",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_data = datasets.MNIST(root = './data', train=True, download=True, transform = transforms.ToTensor())\n",
    "test_data = datasets.MNIST(root = './data', train=False, download=True, transform=transforms.ToTensor())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "a729866f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "60000\n",
      "10000\n"
     ]
    }
   ],
   "source": [
    "print(len(train_data))\n",
    "print(len(test_data))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "cc23cc99",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAaEAAAGxCAYAAADLfglZAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy88F64QAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAeoklEQVR4nO3df2xV9f3H8deVwgWxva5ie1uB2iHIFMQICDTIDweFJjChGBF1K9licAKR4I+IBCnuKzUoxBGUMWMqOFGMQ8TJhBpoYTIYMFQEQ1CK1NHa0UFbCpaVfr5/MG68tvw413t598fzkXwS7rmfN+fd47EvPvfce67POecEAICBK6wbAAC0XoQQAMAMIQQAMEMIAQDMEEIAADOEEADADCEEADBDCAEAzBBCAAAzhBBapddee00+n0+HDh3yXJubmyufz6ejR49GrZ9zf2c0nfsZGxtlZWVR3RcQqTjrBgDEVn5+vnr27Bm27ZprrjHqBghHCAEtXK9evdSvXz/rNoBG8XIc8D8FBQW666671LlzZ7Vv31433HCDpkyZct6X3UpKSpSdna2EhAQFAgE98MAD+ve//91g3qpVqzRo0CB17NhRV111lUaNGqXdu3fH+scBmgVCCPifr776SoMGDdLSpUu1YcMGPf3009q+fbsGDx6s//73vw3mjx8/XjfccIPeeecd5ebmas2aNRo1alTY3Pnz52vSpEm66aab9Pbbb+v1119XdXW17rjjDu3bt89zj4cOHZLP59PkyZMvuWbMmDFq06aNEhMTlZ2drc8//9zzfoFY4eU44H8eeuih0J+dc8rIyNCwYcOUlpamv/71r/rFL34RNj87O1sLFiyQJGVmZio5OVn333+/3n77bd1///0qKSnR3LlzNW3aNC1evDhUN3LkSHXv3l3z5s3TqlWrPPXo8/nUpk0btWnT5qJzg8GgZs+erYEDByohIUF79uzRc889p4EDB+rjjz9Wnz59PO0biAVWQsD/lJeX66GHHlKXLl0UFxentm3bKi0tTZL0xRdfNJh///33hz2+5557FBcXp02bNkmS1q9fr7q6Ov3qV79SXV1daLRv315Dhw5VYWGh5x7T0tJUV1enV1999aJzR48erf/7v//TmDFjNGTIEE2dOlVbtmyRz+fT008/7XnfQCywEgIk1dfXKzMzU0eOHNGcOXPUu3dvdezYUfX19Ro4cKBOnTrVoCYYDIY9jouL0zXXXKOKigpJ0rfffitJ6t+/f6P7vOKKy/9vwOuvv16DBw/Wtm3bLvu+gcYQQoCkzz//XJ9++qlee+015eTkhLZ/+eWX560pKyvTddddF3pcV1enioqK0NufO3XqJEl65513QiuqpsA5ZxKAQGMIIUAKfVDU7/eHbV+2bNl5a9544w317ds39Pjtt99WXV2dhg0bJkkaNWqU4uLi9NVXX2nChAnRbzoCxcXF+vjjjzVixAjrVgBJhBAgSerZs6e6deumJ598Us45JSYm6v3331dBQcF5a1avXq24uDiNHDlSe/fu1Zw5c9SnTx/dc889ks6+9PXMM89o9uzZOnjwoEaPHq2f/OQn+vbbb/WPf/xDHTt21Lx58zz1+fXXX6tbt27Kycm56HWhESNGaMiQIbrllltCb0xYsGCBfD6ffve733naLxArhBAgqW3btnr//ff1yCOPaMqUKYqLi9OIESP00UcfqWvXro3WrF69Wrm5uVq6dKl8Pp/Gjh2rF198Ue3atQvNmTVrlm666Sb9/ve/15tvvqna2loFg0H1798/7N14l8o5pzNnzujMmTMXndu7d2+tWrVKL7zwgk6dOqWkpCTdeeedmjNnjnr06OF530As+JxzzroJAEDrxNVJAIAZQggAYIYQAgCYIYQAAGYIIQCAGUIIAGCmyX1OqL6+XkeOHFF8fHzUv+4YABB7zjlVV1crNTX1oreIanIhdOTIEXXp0sW6DQDAj1RSUqLOnTtfcE6TezkuPj7eugUAQBRcyu/zmIXQyy+/rPT0dLVv3159+/bVli1bLqmOl+AAoGW4lN/nMQmhVatWacaMGZo9e7Z2796tO+64Q1lZWTp8+HAsdgcAaKZicu+4AQMG6LbbbtPSpUtD2372s59p3LhxysvLu2BtVVWVAoFAtFsCAFxmlZWVSkhIuOCcqK+ETp8+rV27dikzMzNse2ZmprZu3dpgfm1traqqqsIGAKB1iHoIHT16VGfOnFFycnLY9uTkZJWVlTWYn5eXp0AgEBq8Mw4AWo+YvTHhhxeknHONXqSaNWuWKisrQ6OkpCRWLQEAmpiof06oU6dOatOmTYNVT3l5eYPVkXT265R/+JXKAIDWIeoroXbt2qlv374Nvha5oKBAGRkZ0d4dAKAZi8kdE2bOnKlf/vKX6tevnwYNGqQ//vGPOnz4cERfZwwAaLliEkITJ05URUWFnnnmGZWWlqpXr15at26d0tLSYrE7AEAzFZPPCf0YfE4IAFoGk88JAQBwqQghAIAZQggAYIYQAgCYIYQAAGYIIQCAGUIIAGCGEAIAmCGEAABmCCEAgBlCCABghhACAJghhAAAZgghAIAZQggAYIYQAgCYIYQAAGYIIQCAGUIIAGCGEAIAmCGEAABmCCEAgBlCCABghhACAJghhAAAZgghAIAZQggAYIYQAgCYIYQAAGYIIQCAGUIIAGCGEAIAmCGEAABmCCEAgBlCCABghhACAJghhAAAZgghAIAZQggAYIYQAgCYIYQAAGYIIQCAGUIIAGCGEAIAmCGEAABmCCEAgBlCCABghhACAJghhAAAZgghAIAZQggAYIYQAgCYibNuAGhK2rRp47kmEAjEoJPomDZtWkR1V155peeaG2+80XPN1KlTPde88MILnmsmTZrkuUaSvvvuO881zz33nOeaefPmea5pKVgJAQDMEEIAADNRD6Hc3Fz5fL6wEQwGo70bAEALEJNrQjfffLM++uij0ONIXmcHALR8MQmhuLg4Vj8AgIuKyTWhAwcOKDU1Venp6br33nt18ODB886tra1VVVVV2AAAtA5RD6EBAwZoxYoVWr9+vV555RWVlZUpIyNDFRUVjc7Py8tTIBAIjS5dukS7JQBAExX1EMrKytKECRPUu3dvjRgxQh988IEkafny5Y3OnzVrliorK0OjpKQk2i0BAJqomH9YtWPHjurdu7cOHDjQ6PN+v19+vz/WbQAAmqCYf06otrZWX3zxhVJSUmK9KwBAMxP1EHrsscdUVFSk4uJibd++XXfffbeqqqqUk5MT7V0BAJq5qL8c980332jSpEk6evSorr32Wg0cOFDbtm1TWlpatHcFAGjmoh5Cb731VrT/SjRRXbt29VzTrl07zzUZGRmeawYPHuy5RpKuvvpqzzUTJkyIaF8tzTfffOO5ZvHixZ5rxo8f77mmurrac40kffrpp55rioqKItpXa8W94wAAZgghAIAZQggAYIYQAgCYIYQAAGYIIQCAGUIIAGCGEAIAmCGEAABmCCEAgBlCCABghhACAJjxOeecdRPfV1VVpUAgYN1Gq3LrrbdGVLdx40bPNfy3bR7q6+s91/z617/2XHPixAnPNZEoLS2NqO7Y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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "image, label = train_data[0]\n",
    "plt.imshow(image.squeeze().numpy(), cmap='gray')\n",
    "plt.title('label : %s' % label)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "a6c05f95",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "torch.Size([1, 28, 28])\n"
     ]
    }
   ],
   "source": [
    "print(image.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "89a3c885",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "name            | type                      | size\n",
      "Num of Batch    |                           | 1200\n",
      "first_batch     | <class 'list'>            | 2\n",
      "first_batch[0]  | <class 'torch.Tensor'>    | torch.Size([50, 1, 28, 28])\n",
      "first_batch[1]  | <class 'torch.Tensor'>    | torch.Size([50])\n"
     ]
    }
   ],
   "source": [
    "#### -- 2-3. Mini-Batch 구성하기 -- ####\n",
    "train_loader = torch.utils.data.DataLoader(dataset = train_data, \n",
    "                                           batch_size = batch_size, shuffle = True)\n",
    "test_loader  = torch.utils.data.DataLoader(dataset = test_data, \n",
    "                                           batch_size = batch_size, shuffle = True)\n",
    "\n",
    "first_batch = train_loader.__iter__().__next__()\n",
    "print('{:15s} | {:<25s} | {}'.format('name', 'type', 'size'))\n",
    "print('{:15s} | {:<25s} | {}'.format('Num of Batch', '', len(train_loader)))\n",
    "print('{:15s} | {:<25s} | {}'.format('first_batch', str(type(first_batch)), len(first_batch)))\n",
    "print('{:15s} | {:<25s} | {}'.format('first_batch[0]', str(type(first_batch[0])), first_batch[0].shape))\n",
    "print('{:15s} | {:<25s} | {}'.format('first_batch[1]', str(type(first_batch[1])), first_batch[1].shape))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c5b47020",
   "metadata": {},
   "outputs": [],
   "source": [
    "class CNN(nn.Module):\n",
    "    def __init__(self):\n",
    "        super(CNN, self).__init__()\n",
    "        self.conv1 = nn.Conv2d(1, 32, 3, 1)\n",
    "        self.conv2 = nn.Conv2d(32, 64, 3, 1)\n",
    "        self.dropout1 = nn.Dropout2d(0.25)\n",
    "        self.dropout2 = nn.Dropout2d(0.5)\n",
    "        self.fc1 = nn.Linear(9216, 128)\n",
    "        self.fc2 = nn.Linear(128, 10)\n",
    "        \n",
    "    def forward(self, x):\n",
    "        x = self.conv1(x)\n",
    "        x = F.relu(x)\n",
    "        x = self.conv2(x)\n",
    "        x = F.max_pool2d(x, 2)\n",
    "        x = self.dropout1(x)\n",
    "        x = torch.flatten(x, 1)\n",
    "        x = self.fc1(x)\n",
    "        x = F.relu(x)\n",
    "        x = self.dropout2(x)\n",
    "        x = self.fc2(x)\n",
    "        output = F.log_softmax(x, dim= 1)\n",
    "        return output\n",
    "    \n",
    "    "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3.9.16 ('test': conda)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.16"
  },
  "vscode": {
   "interpreter": {
    "hash": "26af9df6eef581affd1a85044313b1dcb673309d705ad36f62c5b5e5e3e1a9d3"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
