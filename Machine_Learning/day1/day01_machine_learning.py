from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import jieba
import pandas as pd


def datasets_demo():
    """
    sklearn数据集使用
    :return:
    """
    # 获取数据集
    iris = load_iris()
    print("鸢尾花数据集：\n", iris)
    print("查看数据集描述：\n", iris["DESCR"])
    print("查看特征值的名字：\n", iris.feature_names)
    print("查看特征值：\n", iris.data, iris.data.shape)
    print("查看目标值：\n", iris.target, iris.target.shape)

    # 数据集划分 test_size=0.2测试集比例 random_state种子设置
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("训练集的特征值：\n", x_train, x_train.shape)
    print("测试集的特征值：\n", x_test, x_test.shape)
    print("训练集的目标值：\n", y_train, y_train.shape)
    print("测试集的目标值：\n", y_test, y_test.shape)

    return None


def dict_demo():
    """
    字典特征抽取
    :return:
    """
    data = [{'city': '北京','temperature':100}, {'city': '上海','temperature':60}, {'city': '深圳','temperature':30}]
    # 1、实例化一个转换器类
    transfer = DictVectorizer(sparse=True)

    # 2、调用fit_transform()
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new.toarray(), type(data_new))
    print("特征名字：\n", transfer.get_feature_names_out())

    return None


def count_demo():
    """
    文本特征抽取：CountVecotrizer
    :return:
    """
    data = ["life is short,i like like python", "life is too long,i dislike python"]
    # 1、实例化一个转换器类
    transfer = CountVectorizer(stop_words=["is", "too"])

    # 2、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new.toarray())
    print("特征名字：\n", transfer.get_feature_names())

    return None

def count_chinese_demo():
    """
    中文文本特征抽取：CountVecotrizer
    :return:
    """
    data = ["我 爱 北京 天安门", "天安门 上 太阳 升"]
    # 1、实例化一个转换器类
    transfer = CountVectorizer()

    # 2、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new.toarray())
    print("特征名字：\n", transfer.get_feature_names())

    return None


def cut_word(text):
    """
    进行中文分词："我爱北京天安门" --> "我 爱 北京 天安门"
    :param text:
    :return:
    """
    return " ".join(list(jieba.cut(text)))


def count_chinese_demo2():
    """
    中文文本特征抽取，自动分词
    :return:
    """
    # 将中文文本进行分词
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]

    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    # print(data_new)
    # 1、实例化一个转换器类
    transfer = CountVectorizer(stop_words=["一种", "所以"])

    # 2、调用fit_transform
    data_final = transfer.fit_transform(data_new)
    print("data_new:\n", data_final.toarray())
    print("特征名字：\n", transfer.get_feature_names())

    return None

def tfidf_demo():
    """
    用TF-IDF的方法进行文本特征抽取
    :return:
    """
    # 将中文文本进行分词
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]

    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    # print(data_new)
    # 1、实例化一个转换器类
    transfer = TfidfVectorizer(stop_words=["一种", "所以"])

    # 2、调用fit_transform
    data_final = transfer.fit_transform(data_new)
    print("data_new:\n", data_final.toarray())
    print("特征名字：\n", transfer.get_feature_names())

    return None

def minmax_demo():
    """
    归一化
    :return:
    """
    # 1、获取数据
    data = pd.read_csv("Machine_Learning/day1/dating.txt")
    data = data.iloc[:, :3]
    print("data:\n", data)

    # 2、实例化一个转换器类
    transfer = MinMaxScaler(feature_range=[2, 3]) # 默认feature_range=[0,1]

    # 3、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)

    return None


def stand_demo():
    """
    标准化
    :return:
    """
    # 1、获取数据
    data = pd.read_csv("dating.txt")
    data = data.iloc[:, :3]
    print("data:\n", data)

    # 2、实例化一个转换器类
    transfer = StandardScaler()

    # 3、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    return None

def variance_demo():
    """
    过滤低方差特征
    :return:
    """
    # 1、获取数据
    data = pd.read_csv("Machine_Learning/day1/factor_returns.csv")
    data = data.iloc[:, 1:-2] # 去掉最后两列目标值
    print("data:\n", data)

    # 2、实例化一个转换器类
    transfer = VarianceThreshold(threshold=10) # 越小越苛刻

    # 3、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new, data_new.shape)

    # 计算某两个变量之间的相关系数
    """
statistic: 这是皮尔逊相关系数的值，范围从-1到1。在您的例子中，它是-0.004389322779936285，表示data["pe_ratio"]和data["pb_ratio"]两个数据序列之间的相关性非常弱且接近于无。值为负表示它们之间的关系呈微弱的负相关，即一个变量的增加倾向于与另一个变量的减少相关联，但由于相关性接近于0，这种关系极其微弱。
pvalue: 这是检验统计量的p值，它提供了相关性显著性的度量。在您的例子中，p值是0.8327205496590723，非常接近于1。在统计学中，p值用于判断结果是否具有统计学意义。通常，如果p值小于0.05（或者您选择的显著性水平），则结果被认为是统计学上显著的，即不太可能是由随机变异引起的。在您的案例中，p值远大于0.05，意味着没有足够的证据拒绝零假设，零假设通常是指两个变量之间不存在相关性。因此，您可以得出结论，data["pe_ratio"]和data["pb_ratio"]之间的相关性很可能不具有统计学意义。
简而言之，PearsonRResult输出的两个值告诉您，两个变量之间几乎没有相关性，且这一发现在统计学上不具有显著性。
    """
    r1 = pearsonr(data["pe_ratio"], data["pb_ratio"])
    print("相关系数：\n", r1)
    r2 = pearsonr(data['revenue'], data['total_expense'])
    print("revenue与total_expense之间的相关性：\n", r2)

    return None


def pca_demo():
    """
    PCA降维
    :return:
    """
    data = [[2,8,4,5], [6,3,0,8], [5,4,9,1]]

    # 1、实例化一个转换器类
    # transfer = PCA(n_components=2) # 降维到2维（减少到两个特征）
    transfer = PCA(n_components=0.95) # 保留原有数据95%的特征（信息） 运行后降到了两个特征 意思是我能在降低两个维度的前提下保留原有95%的信息 非常不错

    # 2、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    return None

if __name__ == "__main__":
    # 代码1：sklearn数据集使用
    # datasets_demo()
    # 代码2：字典特征抽取
    # dict_demo()
    # 代码3：文本特征抽取：CountVecotrizer
    # count_demo()
    # 代码4：中文文本特征抽取：CountVecotrizer
    # count_chinese_demo()
    # 代码5：中文文本特征抽取，自动分词
    # count_chinese_demo2()
    # 代码6：中文分词
    # print(cut_word("我爱北京天安门"))
    # 代码7：用TF-IDF的方法进行文本特征抽取
    # tfidf_demo()
    # 代码8：归一化
    # minmax_demo()
    # 代码9：标准化
    # stand_demo()
    # 代码10：低方差特征过滤
    # variance_demo()
    # 代码11：PCA降维
    pca_demo()