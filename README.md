# mljar-api-python

A simple python wrapper over mljar API. It allows MLJAR users to create Machine Learning models with few lines of code:

```python
import mljar

model = Mljar(project='My awesome project', experiment='First experiment')
model.fit(X,y)

model.predict(X)
```

That's all folks! Yeah, I know, this makes Machine Learning super easy! You can use this code for following Machine Learning tasks:
 * Binary classification (your target has only two unique values)
 * Regression (your target value is continuous)
 * More is coming soon!

## How to install

You can install mljar with **pip**:

    pip install -U mljar

or from source code:

    python setup.py install

## How to use it

 1. Create an account at mljar.com and login.
 2. Please go to your users settings (top, right corner).
 3. Get your token, for example 'exampleexampleexample'.
 4. Set environment variable `MLJAR_TOKEN` with your token value:
```
export MLJAR_TOKEN=exampleexampleexample
```
 5. That's all, you are ready to use MLJAR in your python code!

## Examples

Coming soon!
