import os

import tensorflow as tf
from tfx.components import CsvExampleGen
from tfx.components import ExampleValidator
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Transform

import tensorflow_transform.beam as tft_beam
from google.protobuf.json_format import MessageToDict
from tensorflow_transform.tf_metadata import dataset_metadata, schema_utils
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
 
import tempfile
import pprint
import warnings

pp = pprint.PrettyPrinter()

# ignore tf warning messages
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")


# You will define a few global variables to indicate paths in the local workspace.

# location of the pipeline metadata store
_pipeline_root = './pipeline'

# directory of the raw data files
_data_root = './data'

# path to the raw training data
_data_filepath = os.path.join(_data_root, 'metro_traffic_volume.csv')

# ### 1.3 - Preview the  dataset
# 
# The [Metro Interstate Traffic Volume dataset](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume) contains hourly traffic volume of a road in Minnesota from 2012-2018. With this data, you can develop a model for predicting the traffic volume given the date, time, and weather conditions. The attributes are:
#
# Take a quick look at the first few rows of the CSV file.

# ### 1.4 - Create the InteractiveContext
# 
# You will need to initialize the `InteractiveContext` to enable running the TFX components interactively. As before, you will let it create the metadata store in the `_pipeline_root` directory. You can safely ignore the warning about the missing metadata config file.

# Declare the InteractiveContext and use a local sqlite file as the metadata store.
# You can ignore the warning about the missing metadata config file
context = InteractiveContext(pipeline_root=_pipeline_root)

# In the following exercises, you will create the data pipeline components one-by-one, run each of them, and visualize their output artifacts. Recall that we refer to the outputs of pipeline components as *artifacts* and these can be inputs to the next stage of the pipeline.

# The pipeline starts with the [ExampleGen](https://www.tensorflow.org/tfx/guide/examplegen) component. It will:
# 
# *   split the data into training and evaluation sets (by default: 2/3 train, 1/3 eval).
# *   convert each data row into `tf.train.Example` format. This [protocol buffer](https://developers.google.com/protocol-buffers) is designed for Tensorflow operations and is used by the TFX components.
# *   compress and save the data collection under the `_pipeline_root` directory for other components to access. These examples are stored in `TFRecord` format. This optimizes read and write operations within Tensorflow especially if you have a large collection of data.

# Fill out the code below to ingest the data from the CSV file stored in the `_data_root` directory.

### START CODE HERE

# Instantiate ExampleGen with the input CSV dataset
example_gen = CsvExampleGen(input_base=_data_root)

# Run the component using the InteractiveContext instance
context.run(example_gen)

### END CODE HERE


# You should see the output cell of the `InteractiveContext` above showing the metadata associated with the component execution. You can expand the items under `.component.outputs` and see that an `Examples` artifact for the train and eval split is created in `metro_traffic_pipeline/CsvExampleGen/examples/{execution_id}`. 
# 
# You can also check that programmatically with the following snippet. You can focus on the `try` block. The `except` and `else` block is needed mainly for grading. `context.run()` yields no operation when executed in a non-interactive environment (such as the grader script that runs outside of this notebook). In such scenarios, the URI must be manually set to avoid errors.


try:
    # get the artifact object
    artifact = example_gen.outputs['examples'].get()[0]
    
    # print split names and uri
    print(f'split names: {artifact.split_names}')
    print(f'artifact uri: {artifact.uri}')

# for grading since context.run() does not work outside the notebook
except IndexError:
    print("context.run() was no-op")
    examples_path = './pipeline/CsvExampleGen/examples'
    dir_id = os.listdir(examples_path)[0]
    artifact_uri = f'{examples_path}/{dir_id}'

else:
    artifact_uri = artifact.uri


# The ingested data has been saved to the directory specified by the artifact Uniform Resource Identifier (URI). As a sanity check, you can take a look at some of the training examples. This requires working with Tensorflow data types, particularly `tf.train.Example` and `TFRecord` (you can read more about them [here](https://www.tensorflow.org/tutorials/load_data/tfrecord)). Let's first load the `TFRecord` into a variable:

# In[9]:


# Get the URI of the output artifact representing the training examples, which is a directory
train_uri = os.path.join(artifact_uri, 'Split-train')

# Get the list of files in this directory (all compressed TFRecord files)
tfrecord_filenames = [os.path.join(train_uri, name)
                      for name in os.listdir(train_uri)]

# Create a `TFRecordDataset` to read these files
dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")


# <a name='ex-2'></a>
# #### Exercise 2: get_records()
# 
# Complete the helper function below to return a specified number of examples.
# 
# *Hints: You may find the [MessageToDict](https://googleapis.dev/python/protobuf/latest/google/protobuf/json_format.html#google.protobuf.json_format.MessageToDict) helper function and tf.train.Example's [ParseFromString()](https://googleapis.dev/python/protobuf/latest/google/protobuf/message.html#google.protobuf.message.Message.ParseFromString) method useful here. You can also refer [here](https://www.tensorflow.org/tutorials/load_data/tfrecord) for a refresher on TFRecord and tf.train.Example()*

# In[10]:


def get_records(dataset, num_records):
    '''Extracts records from the given dataset.
    Args:
        dataset (TFRecordDataset): dataset saved by ExampleGen
        num_records (int): number of records to preview
    '''
    
    # initialize an empty list
    records = []

    ### START CODE HERE
    # Use the `take()` method to specify how many records to get
    for tfrecord in dataset.take(num_records):
        
        # Get the numpy property of the tensor
        serialized_example = tfrecord.numpy()
        
        # Initialize a `tf.train.Example()` to read the serialized data
        example = tf.train.Example()
        
        # Read the example data (output is a protocol buffer message)
        example.ParseFromString(serialized_example)
        
        # convert the protocol buffer message to a Python dictionary
        example_dict = MessageToDict(example)
        
        # append to the records list
        records.append(example_dict)
        
    ### END CODE HERE
    return records


# In[11]:


# Get 3 records from the dataset
sample_records = get_records(dataset, 3)

# Print the output
pp.pprint(sample_records)


# You should see three of the examples printed above. Now that `ExampleGen` has finished ingesting the data, the next step is data analysis.

# <a name='2-2'></a>
# ### 2.2 - StatisticsGen
# The [StatisticsGen](https://www.tensorflow.org/tfx/guide/statsgen) component computes statistics over your dataset for data analysis, as well as for use in downstream components. It uses the [TensorFlow Data Validation](https://www.tensorflow.org/tfx/data_validation/get_started) library.
# 
# `StatisticsGen` takes as input the dataset ingested using `CsvExampleGen`.

# <a name='ex-3'></a>
# #### Exercise 3: StatisticsGen
# 
# Fill the code below to generate statistics from the output examples of `CsvExampleGen`.

# In[12]:


### START CODE HERE
# Instantiate StatisticsGen with the ExampleGen ingested dataset
statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    

# Run the component
context.run(statistics_gen)
### END CODE HERE


# In[13]:


# Plot the statistics generated
context.show(statistics_gen.outputs['statistics'])


# <a name='2-3'></a>
# ### 2.3 - SchemaGen
# 
# The [SchemaGen](https://www.tensorflow.org/tfx/guide/schemagen) component also uses TFDV to generate a schema based on your data statistics. As you've learned previously, a schema defines the expected bounds, types, and properties of the features in your dataset.
# 
# `SchemaGen` will take as input the statistics that we generated with `StatisticsGen`, looking at the training split by default.

# <a name='ex-4'></a>
# #### Exercise 4: SchemaGen

# In[14]:


### START CODE HERE
# Instantiate SchemaGen with the output statistics from the StatisticsGen
schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])
    
    

# Run the component
context.run(schema_gen)
### END CODE HERE


# If all went well, you can now visualize the generated schema as a table.

# In[15]:


# Visualize the output
context.show(schema_gen.outputs['schema'])


# Each attribute in your dataset shows up as a row in the schema table, alongside its properties. The schema also captures all the values that a categorical feature takes on, denoted as its domain.
# 
# This schema will be used to detect anomalies in the next step.

# <a name='2-4'></a>
# ### 2.4 - ExampleValidator
# 
# The [ExampleValidator](https://www.tensorflow.org/tfx/guide/exampleval) component detects anomalies in your data based on the generated schema from the previous step. Like the previous two components, it also uses TFDV under the hood. 
# 
# `ExampleValidator` will take as input the statistics from `StatisticsGen` and the schema from `SchemaGen`. By default, it compares the statistics from the evaluation split to the schema from the training split.

# <a name='2-4'></a>
# #### Exercise 5: ExampleValidator
# 
# Fill the code below to detect anomalies in your datasets.

# In[16]:


### START CODE HERE
# Instantiate ExampleValidator with the statistics and schema from the previous steps
example_validator = ExampleValidator(statistics=statistics_gen.outputs['statistics'],schema=schema_gen.outputs['schema'])
    
    

# Run the component
context.run(example_validator)
### END CODE HERE


# As with the previous steps, you can visualize the anomalies as a table.

# In[17]:


# Visualize the output
context.show(example_validator.outputs['anomalies'])


# If there are anomalies detected, you should examine how you should handle it. For example, you can relax distribution constraints or modify the domain of some features. You've had some practice with this last week when you used TFDV and you can also do that here. 
# 
# For this particular case, there should be no anomalies detected and we can proceed to the next step.

# <a name='2-5'></a>
# ### 2.5 - Transform
# 
# In this section, you will use the [Transform](https://www.tensorflow.org/tfx/guide/transform) component to perform feature engineering.
# 
# `Transform` will take as input the data from `ExampleGen`, the schema from `SchemaGen`, as well as a module containing the preprocessing function.
# 
# The component expects an external module for your Transform code so you need to use the magic command `%% writefile` to save the file to disk. We have defined a few constants that group the data's attributes according to the transforms you will perform later. This file will also be saved locally.

# In[18]:


# Set the constants module filename
_traffic_constants_module_file = 'traffic_constants.py'


# In[19]:


get_ipython().run_cell_magic('writefile', '{_traffic_constants_module_file}', "\n# Features to be scaled to the z-score\nDENSE_FLOAT_FEATURE_KEYS = ['temp', 'snow_1h']\n\n# Features to bucketize\nBUCKET_FEATURE_KEYS = ['rain_1h']\n\n# Number of buckets used by tf.transform for encoding each feature.\nFEATURE_BUCKET_COUNT = {'rain_1h': 3}\n\n# Feature to scale from 0 to 1\nRANGE_FEATURE_KEYS = ['clouds_all']\n\n# Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform\nVOCAB_SIZE = 1000\n\n# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.\nOOV_SIZE = 10\n\n# Features with string data types that will be converted to indices\nVOCAB_FEATURE_KEYS = [\n    'holiday',\n    'weather_main',\n    'weather_description'\n]\n\n# Features with int data type that will be kept as is\nCATEGORICAL_FEATURE_KEYS = [\n    'hour', 'day', 'day_of_week', 'month'\n]\n\n# Feature to predict\nVOLUME_KEY = 'traffic_volume'\n\ndef transformed_name(key):\n    return key + '_xf'")


# <a name='ex-6'></a>
# #### Exercise 6

# Next, you will fill out the transform module. As mentioned, this will also be saved to disk. Specifically, you will complete the `preprocessing_fn` which defines the transformations. See the code comments for instructions and refer to the [tft module documentation](https://www.tensorflow.org/tfx/transform/api_docs/python/tft) to look up which function to use for a given group of keys.
# 
# For the label (i.e. `VOLUME_KEY`), you will transform it to indicate if it is greater than the mean of the entire dataset.

# In[20]:


# Set the transform module filename
_traffic_transform_module_file = 'traffic_transform.py'


# In[21]:


get_ipython().run_cell_magic('writefile', '{_traffic_transform_module_file}', '\nimport tensorflow as tf\nimport tensorflow_transform as tft\n\nimport traffic_constants\n\n# Unpack the contents of the constants module\n_DENSE_FLOAT_FEATURE_KEYS = traffic_constants.DENSE_FLOAT_FEATURE_KEYS\n_RANGE_FEATURE_KEYS = traffic_constants.RANGE_FEATURE_KEYS\n_VOCAB_FEATURE_KEYS = traffic_constants.VOCAB_FEATURE_KEYS\n_VOCAB_SIZE = traffic_constants.VOCAB_SIZE\n_OOV_SIZE = traffic_constants.OOV_SIZE\n_CATEGORICAL_FEATURE_KEYS = traffic_constants.CATEGORICAL_FEATURE_KEYS\n_BUCKET_FEATURE_KEYS = traffic_constants.BUCKET_FEATURE_KEYS\n_FEATURE_BUCKET_COUNT = traffic_constants.FEATURE_BUCKET_COUNT\n_VOLUME_KEY = traffic_constants.VOLUME_KEY\n_transformed_name = traffic_constants.transformed_name\n\n\ndef preprocessing_fn(inputs):\n    """tf.transform\'s callback function for preprocessing inputs.\n    Args:\n    inputs: map from feature keys to raw not-yet-transformed features.\n    Returns:\n    Map from string feature key to transformed feature operations.\n    """\n    outputs = {}\n\n    ### START CODE HERE\n    \n    # Scale these features to the z-score.\n    for key in _DENSE_FLOAT_FEATURE_KEYS:\n        # Scale these features to the z-score.\n        outputs[_transformed_name(key)] = tft.scale_to_z_score(inputs[key])\n            \n\n    # Scale these feature/s from 0 to 1\n    for key in _RANGE_FEATURE_KEYS:\n        outputs[_transformed_name(key)] = tft.scale_to_0_1(inputs[key])\n            \n\n    # Transform the strings into indices \n    # hint: use the VOCAB_SIZE and OOV_SIZE to define the top_k and num_oov parameters\n    for key in _VOCAB_FEATURE_KEYS:\n        outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(\n            inputs[key], \n            top_k=_VOCAB_SIZE, \n            num_oov_buckets=_OOV_SIZE)\n            \n            \n            \n\n    # Bucketize the feature\n    for key in _BUCKET_FEATURE_KEYS:\n        outputs[_transformed_name(key)] = tft.bucketize(\n            inputs[key], \n            _FEATURE_BUCKET_COUNT[key])\n            \n\n    # Keep the features as is. No tft function needed.\n    for key in _CATEGORICAL_FEATURE_KEYS:\n        outputs[_transformed_name(key)] = inputs[key]\n\n        \n    # Use `tf.cast` to cast the label key to float32\n    traffic_volume = tf.cast(inputs[_VOLUME_KEY], tf.float32)\n  \n    \n    # Create a feature that shows if the traffic volume is greater than the mean and cast to an int\n    outputs[_transformed_name(_VOLUME_KEY)] = tf.cast(  \n        \n        # Use `tf.greater` to check if the traffic volume in a row is greater than the mean of the entire traffic volumn column\n        tf.greater(traffic_volume, tft.mean(tf.cast(inputs[_VOLUME_KEY], tf.float32))),\n        \n        tf.int64)                                        \n\n    ### END CODE HERE\n    return outputs')


# In[22]:


# Test your preprocessing_fn

import traffic_transform
from testing_values import feature_description, raw_data

# NOTE: These next two lines are for reloading your traffic_transform module in case you need to 
# update your initial solution and re-run this cell. Please do not remove them especially if you
# have revised your solution. Else, your changes will not be detected.
import importlib
importlib.reload(traffic_transform)

raw_data_metadata = dataset_metadata.DatasetMetadata(schema_utils.schema_from_feature_spec(feature_description))

with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
    transformed_dataset, _ = (
        (raw_data, raw_data_metadata) | tft_beam.AnalyzeAndTransformDataset(traffic_transform.preprocessing_fn))

transformed_data, transformed_metadata = transformed_dataset


# In[23]:


# Test that the transformed data matches the expected output
transformed_data


# **Expected Output:**
# 
# ```
# [{'clouds_all_xf': 1.0,
#   'day_of_week_xf': 4,
#   'day_xf': 8,
#   'holiday_xf': 0,
#   'hour_xf': 15,
#   'month_xf': 1,
#   'rain_1h_xf': 2,
#   'snow_1h_xf': 0.0,
#   'temp_xf': 0.0,
#   'traffic_volume_xf': 0,
#   'weather_description_xf': 0,
#   'weather_main_xf': 0}]
# ```

# In[24]:


# Test that the transformed metadata's schema matches the expected output
MessageToDict(transformed_metadata.schema)


# **Expected Output:**
# 
# ```
# {'feature': [{'name': 'clouds_all_xf',
#    'type': 'FLOAT',
#    'presence': {'minFraction': 1.0},
#    'shape': {}},
#   {'name': 'day_of_week_xf',
#    'type': 'INT',
#    'presence': {'minFraction': 1.0},
#    'shape': {}},
#   {'name': 'day_xf',
#    'type': 'INT',
#    'presence': {'minFraction': 1.0},
#    'shape': {}},
#   {'name': 'holiday_xf',
#    'type': 'INT',
#    'intDomain': {'isCategorical': True},
#    'presence': {'minFraction': 1.0},
#    'shape': {}},
#   {'name': 'hour_xf',
#    'type': 'INT',
#    'presence': {'minFraction': 1.0},
#    'shape': {}},
#   {'name': 'month_xf',
#    'type': 'INT',
#    'presence': {'minFraction': 1.0},
#    'shape': {}},
#   {'name': 'rain_1h_xf',
#    'type': 'INT',
#    'intDomain': {'isCategorical': True},
#    'presence': {'minFraction': 1.0},
#    'shape': {}},
#   {'name': 'snow_1h_xf',
#    'type': 'FLOAT',
#    'presence': {'minFraction': 1.0},
#    'shape': {}},
#   {'name': 'temp_xf',
#    'type': 'FLOAT',
#    'presence': {'minFraction': 1.0},
#    'shape': {}},
#   {'name': 'traffic_volume_xf',
#    'type': 'INT',
#    'presence': {'minFraction': 1.0},
#    'shape': {}},
#   {'name': 'weather_description_xf',
#    'type': 'INT',
#    'intDomain': {'isCategorical': True},
#    'presence': {'minFraction': 1.0},
#    'shape': {}},
#   {'name': 'weather_main_xf',
#    'type': 'INT',
#    'intDomain': {'isCategorical': True},
#    'presence': {'minFraction': 1.0},
#    'shape': {}}]}
# ```

# <a name='ex-7'></a>
# #### Exercise 7
# 
# With the transform module defined, complete the code below to perform feature engineering on the raw data.

# In[25]:


### START CODE HERE
# Instantiate the Transform component
transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=os.path.abspath(_traffic_transform_module_file))
    
    
    

# Run the component.
# The `enable_cache` flag is disabled in case you need to update your transform module file.
context.run(transform, enable_cache=False)
### END CODE HERE


# You should see the output cell by `InteractiveContext` above and see the three artifacts in `.component.outputs`:
# 
# * `transform_graph` is the graph that performs the preprocessing operations. This will be included during training and serving to ensure consistent transformations of incoming data.
# * `transformed_examples` points to the preprocessed training and evaluation data.
# * `updated_analyzer_cache` are stored calculations from previous runs.

# The `transform_graph` artifact URI should point to a directory containing:
# 
# * The `metadata` subdirectory containing the schema of the original data.
# * The `transformed_metadata` subdirectory containing the schema of the preprocessed data. 
# * The `transform_fn` subdirectory containing the actual preprocessing graph.
# 
# Again, for grading purposes, we inserted an `except` and `else` below to handle checking the output outside the notebook environment.

# In[26]:


try:
    # Get the uri of the transform graph
    transform_graph_uri = transform.outputs['transform_graph'].get()[0].uri

except IndexError:
    print("context.run() was no-op")
    transform_path = './pipeline/Transform/transformed_examples'
    dir_id = os.listdir(transform_path)[0]
    transform_graph_uri = f'{transform_path}/{dir_id}'
    
else:
    # List the subdirectories under the uri
    os.listdir(transform_graph_uri)


# Lastly, you can also take a look at a few of the transformed examples.

# In[27]:


try:
    # Get the URI of the output artifact representing the transformed examples
    train_uri = os.path.join(transform.outputs['transformed_examples'].get()[0].uri, 'Split-train')
    
except IndexError:
    print("context.run() was no-op")
    train_uri = os.path.join(transform_graph_uri, 'Split-train')


# In[28]:


# Get the list of files in this directory (all compressed TFRecord files)
tfrecord_filenames = [os.path.join(train_uri, name)
                      for name in os.listdir(train_uri)]

# Create a `TFRecordDataset` to read these files
transformed_dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")


# In[29]:


# Get 3 records from the dataset
sample_records_xf = get_records(transformed_dataset, 3)

# Print the output
pp.pprint(sample_records_xf)


# **Congratulations on completing this week's assignment!** You've just demonstrated how to build a data pipeline and do feature engineering. You will build upon these concepts in the next weeks where you will deal with more complex datasets and also access the metadata store. Keep up the good work!

# In[ ]:


# grade-up-to-here

