from IPython.display import Markdown, display
def printmd(string):
    display(Markdown('# <span style="color:red">'+string+'</span>'))


if ('sc' in locals() or 'sc' in globals()):
    printmd('<<<<<!!!!! It seems that you are running in a IBM Watson Studio Apache Spark Notebook. Please run it in an IBM Watson Studio Default Runtime (without Apache Spark) !!!!!>>>>>')

#pip install pyspark==2.4.5

try:
    from pyspark import SparkContext, SparkConf
    from pyspark.sql import SparkSession
except ImportError as e:
    printmd('<<<<<!!!!! Please restart your kernel after installing Apache Spark !!!!!>>>>>')

sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

spark = SparkSession \
    .builder \
    .getOrCreate()

rdd = sc.parallelize(range(100))

rdd.count()

rdd.sum()

def gt50(i):
    if i > 50:
        return True
    else:
        return False

print(gt50(4))
print(gt50(51))

def gt50(i):
    return i > 50

print(gt50(4))
print(gt50(51))

gt50 = lambda i: i > 50

print(gt50(4))
print(gt50(51))

#let's shuffle our list to make it a bit more interesting
from random import shuffle
l = list(range(100))
shuffle(l)
rdd = sc.parallelize(l)

rdd.filter(gt50).collect()

rdd.filter(lambda i: i > 50).collect()

rdd.filter(lambda x: x>50).filter(lambda x: x<75).sum() # 1500

from pyspark.sql import Row

df = spark.createDataFrame([Row(id=1, value='value1'),Row(id=2, value='value2')])

# let's have a look what's inside
df.show()

# let's print the schema
df.printSchema()

# register dataframe as query table
df.createOrReplaceTempView('df_view')

# execute SQL query
df_result = spark.sql('select value from df_view where id=2')

# examine contents of result
df_result.show()

# get result as string
df_result.first().value

df.count()

