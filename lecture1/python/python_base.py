# Desc: python基础语法
# 数据结构
## 1. 布尔型
a = True
b = False

## 2. 数值型
x = 2
y = 1.0
z = x / y

## 3. 字符串
str1 = 'This is Dr.Wu'
str2 = "I'm Dr.Wu"
str3 = "Hello! everyone, I'm Dr.Wu"
s1 = '*'
s_b = ' '
## 3.1 字符串输出模式
print(s1 * 20)
print(str1 + ', ' + str2)
s2 = 'name kind'
s2.upper()
s2.lower()
s2.title()
print(s1 * 20)
print(" "+ s1 * 18)
print("  "+ s1 * 16)

for i in range(1, 11):
    print(f'{s_b * i}{s1* (21 - i*2)}')

## 3.2 字符串输出引用
st1 = 'Allen'
st2 = 'happy'
print("%s is %s" % (st1, st2))
print("{} is {}".format(st1, st2))
print(f"{st1} is {st2}")

## 4. list
c = []
c.append(1)
c.extend([2, 3, 4])
c.append(5)
c.extend([2, 3, 4])
c.index(2)
c.insert(1, 0)
c.pop(2)
c.reverse()
c.sort()
c.remove(2)

## 5. tuple
d = (1, 'two', 3)

## 6. dict
e = {'1': 'one', 2: 'two', '3': 'three'}
e.keys()
e.values()
e.update({'3': '3########'})

## 7. set
f = set([1, 2, 3, 4, 5,2,3]) # 集合
g = set([1, 2, 3,6])
f-g # 差集
f|g # 并集
f&g # 交集

# if else
a = 5
if a > 10:
    print('a > 10')
elif a == 10:
    print('a = 10')
else:
    print('a < 10')

# for loop 计算1+2+3+...+100
sum = 0
for i in range(0, 101,2):
    print(i)
    sum +=  i
print(f"1+2+3+...+100 = {sum}")