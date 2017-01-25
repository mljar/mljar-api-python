from mljar_client import MljarClient

from mljar import Mljar
import pandas as pd
import numpy as np

# load example data file
fname = '/home/piotr/webs/mljar/test/data/binary_part_iris_converted.csv'
df = pd.read_csv(fname)
cols = ['sepal length', 'sepal width', 'petal length', 'petal width']

# create MLJAR models, yeah! :)
model = Mljar(project = 'Iris_binary', experiment = 'Experiment_1', verbose = True)
model.fit(X = df[cols], y = df['class'])

# predict with MLJAR model
predictions = model.predict(df[cols])

print 'Super!'
