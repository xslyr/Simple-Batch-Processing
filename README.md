# Simple-Batch-Processing
Python library to help you slice the heavy processing of your machine learning models.

  - import : simplebatchp
  - class : SimpleBatchP
    - paramenters on creating object:
      model: function that will receive batch processing
      x: numpy array with dependent variable
      y: numpy array with independent variable (optional)
    - method do_sliced_task: 
      blocks_of_execution: number of times that our model will be run
      model_params: possible parameter that can imput on function (optional)
      
Example usage:
```
from simplebatchp import SimpleBatchP
import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]], dtype=int)
y = np.array([2,4,6,8,10,12,14,16,18,20], dtype=int)

regressor = LinearRegression()

# regressor.fit( x , y ) is the common used for non batch processing
# below is how our library can be used

my_batch_processor = SimpleBatchP( model=regressor.fit, x, y )
my_batch_processor.do_sliced_task( blocks_of_execution=2 )

```
