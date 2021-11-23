# Filtering conditions

## Filter by length/duration

Periods within a condition can be filtered on duration using the *filter* method

```python
from perambulate import Condition
from perambulate.datasets import load_sinusoid


df = load_sinusoid()

A = Condition((df.sinusoid > -0.5) & (df.sinusoid < 0.75))
A.plot(df)
```

![image](https://user-images.githubusercontent.com/47210270/143017250-c2019a87-3cb6-40a4-8fc9-6b879f68a5f1.png)


```python
>>> B = A.filter(">3d").filter("<4d")
>>> B.plot(df)
```

![image](https://user-images.githubusercontent.com/47210270/143017293-0786fcce-43fd-4303-ba02-08609eb6f6e9.png)
