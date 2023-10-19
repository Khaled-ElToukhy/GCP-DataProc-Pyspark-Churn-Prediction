# %% [markdown]
# Importing Modules

# %%
pip install psycopg2-binary

# %%
# importing spark session
from pyspark.sql import SparkSession

# data visualization modules 
import matplotlib.pyplot as plt
import seaborn as sns

from sqlalchemy import create_engine
import psycopg2

# pandas module 
import pandas as pd

# pyspark SQL functions 
from pyspark.sql.functions import col, when, count, udf

# pyspark data preprocessing modules
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler, StandardScaler, OneHotEncoder

# pyspark data modeling and model evaluation modules
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from IPython.core.display import display, HTML

# %%
display(HTML("<style>pre { white-space: pre !important; }</style>"))

# %% [markdown]
# Building our Spark Session

# %%
spark = SparkSession.builder.appName("Customer_Churn_Prediction").getOrCreate()
spark

# %% [markdown]
# Loading our data

# %%
data = spark.read.csv("dataset.csv",inferSchema=True,header=True)
data.show(4)

# %% [markdown]
# Print the data schema to check out the data types

# %%
data.printSchema()

# %% [markdown]
# Get the data dimension 

# %%
data.count()

# %%
len(data.columns)

# %%
data.dtypes

# %% [markdown]
# Let's get all the numerical features and store them into a pandas dataframe.

# %%
numerical_columns = [name for name, typ in data.dtypes if typ == 'int' or typ == 'double']
categorical_columns = [name for name, typ in data.dtypes if typ == 'string']

data.select(categorical_columns).show(10)

# %%
df = data.select(numerical_columns).toPandas()
df.head()

# %% [markdown]
# Let's create histograms to analyse the distribution of our numerical columns. 

# %%
fig = plt.figure(figsize=(15,6))
ax =fig.gca()
df.hist(bins =20,ax=ax)
plt.tight_layout()
plt.show()

# %% [markdown]
# Let's generate the correlation matrix 

# %%
df.corr()

# %% [markdown]
# Let's check the unique value count per each categorical variables

# %%
for column in categorical_columns:
    data.groupby(column).count().show()

# %% [markdown]
# Let's find number of null values in all of our dataframe columns

# %%
for column in data.columns:
    data.select(count(when(col(column).isNull(),column)).alias(column)).show()

# %%
missing_col = ['TotalCharges']

# %%
imputer = Imputer(inputCols = missing_col , outputCols= missing_col).setStrategy('mean')

# %%
imputer = impute.fit(data)

# %%
data = imputer.transform(data)
data.select(count(when(col('TotalCharges').isNull(),'TotalCharges')).alias('TotalCharges')).show()

# %%
data.select("*").where(data.tenure > 100).show()

# %%
data = data.filter(data.tenure < 100)

# %%
numerical_vector_assembler = VectorAssembler(inputCols= numerical_columns, outputCol="numerical_vector_assembled")
data = numerical_vector_assembler.transform(data)
data.show(5)

# %%
scaler = StandardScaler(inputCol= "numerical_vector_assembled", outputCol="numerical_vector_scaled", withStd=True, withMean=True)
data = scaler.fit(data).transform(data)
data.show(10)

# %%
categorical_columns_indexed = [ name + "_indexed" for name in categorical_columns]
indexer = StringIndexer(inputCols=categorical_columns,outputCols= categorical_columns_indexed)
data = indexer.fit(data).transform(data)
data.show(10)

# %% [markdown]
# Let's combine all of our categorifal features in to one feature vector.

# %%
categorical_columns_indexed.remove('customerID_indexed')
categorical_columns_indexed.remove('Churn_indexed')


categorical_vector_assembler = VectorAssembler(inputCols= categorical_columns_indexed, outputCol="categorical_features_assembled")
data = categorical_vector_assembler.transform(data)
data.show(5)

# %% [markdown]
# Now let's combine categorical and numerical feature vectors.

# %%
scaler = StandardScaler(inputCol= "categorical_features_assembled", outputCol="categorical_features_scaled", withStd=True, withMean=True)
data = scaler.fit(data).transform(data)
data.show(10)

# %%
final_vector_assembler = VectorAssembler(inputCols=['categorical_features_scaled','numerical_vector_scaled'], outputCol= 'final_feature_vectored')
data = final_vector_assembler.transform(data)
data.show(10)

# %%
train ,test = data.randomSplit([0.7,0.3], seed = 0)

# %%
print(train.count())
print(test.count())

# %% [markdown]
# Now let's create and train our desicion tree

# %%
dt = DecisionTreeClassifier(featuresCol= "final_feature_vectored" , labelCol= 'Churn_indexed')

# %%
model = dt.fit(train)

# %% [markdown]
# Let's make predictions on our test data

# %%
test_pred = model.transform(test)
test_pred.select(['churn','prediction']).show()

# %%
evaluator = BinaryClassificationEvaluator(labelCol="Churn_indexed")
auc_test = evaluator.evaluate(test_pred)
auc_test

# %% [markdown]
# Let's get the AUC for our `training` set

# %%
predictions_train = model.transform(train)
auc_train = evaluator.evaluate(predictions_train)
auc_train

# %%
train.show(10)

# %%
test.show(10)

# %% [markdown]
# **Hyper parameter tuning** 

# %%
def evaluate_dt(mode_params):
      test_accuracies = []
      train_accuracies = []

      for maxD in mode_params:
        # train the model based on the maxD
        decision_tree = DecisionTreeClassifier(featuresCol = 'final_feature_vectored', labelCol ='Churn_indexed', maxDepth = maxD)
        dtModel = decision_tree.fit(train)

        # calculating test error 
        predictions_test = dtModel.transform(test)
        evaluator = BinaryClassificationEvaluator(labelCol="Churn_indexed")
        auc_test = evaluator.evaluate(predictions_test, {evaluator.metricName: "areaUnderROC"})
        # recording the accuracy 
        test_accuracies.append(auc_test)

        # calculating training error
        predictions_training = dtModel.transform(train)
        evaluator = BinaryClassificationEvaluator(labelCol="Churn_indexed")
        auc_training = evaluator.evaluate(predictions_training, {evaluator.metricName: "areaUnderROC"})
        train_accuracies.append(auc_training)

      return(test_accuracies, train_accuracies)  

# %% [markdown]
# Let's define `params` list to evaluate our model iteratively with differe maxDepth parameter.  

# %%
max_depth = [5,10,15,20]
test_accs  , train_accs = evaluate_dt(max_depth)
print(train_accs)
print(test_accs)

# %% [markdown]
# Let's visualize our results

# %%
df = pd.DataFrame()

df['max_depth'] = max_depth
df['train_accs'] = train_accs
df['test_accs'] = test_accs


sns.lineplot(df , x="max_depth" , y='train_accs')

# %%
sns.lineplot(df , x="max_depth" , y='test_accs')


