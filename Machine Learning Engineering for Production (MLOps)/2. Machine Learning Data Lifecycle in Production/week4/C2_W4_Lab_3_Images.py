import os
import pprint
import tempfile
import urllib

import absl
import tensorflow as tf
tf.get_logger().propagate = False
pp = pprint.PrettyPrinter()

from tfx import v1 as tfx

from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx.types import Channel

from google.protobuf.json_format import MessageToDict

print('TensorFlow version: {}'.format(tf.__version__))
print('TFX version: {}'.format(tfx.__version__))


# ## Set up pipeline paths

# In[2]:


# Location of the pipeline metadata store
_pipeline_root = './pipeline/'

# Data files directory
_data_root = './data/cifar10'

# Path to the training data
_data_filepath = os.path.join(_data_root, 'train.tfrecord')


# ## Download example data
# 
# We will download the training split of the CIFAR-10 dataset and save it to the `_data_filepath`. Take note that this is already in TFRecord format so we won't need to convert it when we use `ExampleGen` later.

# In[3]:


# Create data folder for the images
# get_ipython().system('mkdir -p {_data_root}')

# URL of the hosted dataset
DATA_PATH = 'https://raw.githubusercontent.com/tensorflow/tfx/v0.21.4/tfx/examples/cifar10/data/train.tfrecord'

# Download the dataset and save locally
urllib.request.urlretrieve(DATA_PATH, _data_filepath)


# ## Create the InteractiveContext

# In[4]:


# Initialize the InteractiveContext
context = InteractiveContext(pipeline_root=_pipeline_root)


# ## Run TFX components interactively
# 
# 

# ### ExampleGen
# 
# As mentioned earlier, the dataset is already in TFRecord format so, unlike the previous TFX labs, there is no need to convert it when we ingest the data. You can simply import it with [ImportExampleGen](https://www.tensorflow.org/tfx/api_docs/python/tfx/components/ImportExampleGen) and here is the syntax and modules for that.

# In[5]:


# Ingest the data through ExampleGen
example_gen = tfx.components.ImportExampleGen(input_base=_data_root)

# Run the component
context.run(example_gen)


# As usual, this component produces two artifacts, training examples and evaluation examples:

# In[6]:


# Print split names and URI
artifact = example_gen.outputs['examples'].get()[0]
print(artifact.split_names, artifact.uri)


# You can also take a look at the first three training examples ingested by using the `tf.io.parse_single_example()` method from the [tf.io](https://www.tensorflow.org/api_docs/python/tf/io) module. See how it is setup in the cell below.

# In[7]:


import IPython.display as display

# Get the URI of the output artifact representing the training examples, which is a directory
train_uri = os.path.join(example_gen.outputs['examples'].get()[0].uri, 'Split-train')

# Get the list of files in this directory (all compressed TFRecord files)
tfrecord_filenames = [os.path.join(train_uri, name)
                      for name in os.listdir(train_uri)]

# Create a `TFRecordDataset` to read these files
dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")

# Description per example
image_feature_description = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}

# Image parser function
def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)

# Map the parser to the dataset
parsed_image_dataset = dataset.map(_parse_image_function)

# Display the first three images
for features in parsed_image_dataset.take(3):
    image_raw = features['image_raw'].numpy()
    display.display(display.Image(data=image_raw))
    pprint.pprint('Class ID: {}'.format(features['label'].numpy()))


# ### StatisticsGen
# 
# Next, you will generate the statistics so you can infer a schema in the next step. You can also look at the visualization of the statistics. As you might expect with CIFAR-10, there is a column for the image and another column for the numeric label.

# In[8]:


# Run StatisticsGen
statistics_gen = tfx.components.StatisticsGen(
    examples=example_gen.outputs['examples'])

context.run(statistics_gen)


# In[9]:


# Visualize the results
context.show(statistics_gen.outputs['statistics'])


# ### SchemaGen
# 
# Here, you pass in the statistics to generate the Schema. For the version of TFX you are using, you will have to explicitly set `infer_feature_shape=True` so the downstream TFX components (e.g. Transform) will parse input as a `Tensor` and not `SparseTensor`. If not set, you will have compatibility issues later when you run the transform.

# In[10]:


# Run SchemaGen
schema_gen = tfx.components.SchemaGen(
      statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)
context.run(schema_gen)


# In[11]:


# Visualize the results
context.show(schema_gen.outputs['schema'])


# ### ExampleValidator
# 
# `ExampleValidator` is not required but you can still run it just to make sure that there are no anomalies.

# In[12]:


# Run ExampleValidator
example_validator = tfx.components.ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema'])
context.run(example_validator)


# In[13]:


# Visualize the results. There should be no anomalies.
context.show(example_validator.outputs['anomalies'])


# ### Transform
# 
# To successfully transform the raw image, you need to parse the current bytes format and convert it to a tensor. For that, you can use the [tf.image.decode_image()](https://www.tensorflow.org/api_docs/python/tf/io/decode_image) function. The transform module below utilizes this and converts the image to a `(32,32,3)` shaped float tensor. It also scales the pixels and converts the labels to one-hot tensors. The output features should then be ready to pass on to a model that accepts this format.

# In[14]:


_transform_module_file = 'cifar10_transform.py'


# In[15]:


# get_ipython().run_cell_magic('writefile', '{_transform_module_file}', '\nimport tensorflow as tf\nimport tensorflow_transform as tft\n\n# Keys\n_LABEL_KEY = \'label\'\n_IMAGE_KEY = \'image_raw\'\n\n\ndef _transformed_name(key):\n    return key + \'_xf\'\n\ndef _image_parser(image_str):\n    \'\'\'converts the images to a float tensor\'\'\'\n    image = tf.image.decode_image(image_str, channels=3)\n    image = tf.reshape(image, (32, 32, 3))\n    image = tf.cast(image, tf.float32)\n    return image\n\n\ndef _label_parser(label_id):\n    \'\'\'one hot encodes the labels\'\'\'\n    label = tf.one_hot(label_id, 10)\n    return label\n\n\ndef preprocessing_fn(inputs):\n    """tf.transform\'s callback function for preprocessing inputs.\n    Args:\n        inputs: map from feature keys to raw not-yet-transformed features.\n    Returns:\n        Map from string feature key to transformed feature operations.\n    """\n    \n    # Convert the raw image and labels to a float array and\n    # one-hot encoded labels, respectively.\n    with tf.device("/cpu:0"):\n        outputs = {\n            _transformed_name(_IMAGE_KEY):\n                tf.map_fn(\n                    _image_parser,\n                    tf.squeeze(inputs[_IMAGE_KEY], axis=1),\n                    dtype=tf.float32),\n            _transformed_name(_LABEL_KEY):\n                tf.map_fn(\n                    _label_parser,\n                    tf.squeeze(inputs[_LABEL_KEY], axis=1),\n                    dtype=tf.float32)\n        }\n    \n    # scale the pixels from 0 to 1\n    outputs[_transformed_name(_IMAGE_KEY)] = tft.scale_to_0_1(outputs[_transformed_name(_IMAGE_KEY)])\n    \n    return outputs')


# Now, we pass in this feature engineering code to the `Transform` component and run it to transform your data.

# In[16]:


# Ignore TF warning messages
tf.get_logger().setLevel('ERROR')

# Setup the Transform component
transform = tfx.components.Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=os.path.abspath(_transform_module_file))

# Run the component
context.run(transform)


# ### Preview the results
# 
# Now that the Transform component is finished, you can preview how the transformed images and labels look like. You can use the same sequence and helper function from previous labs.

# In[17]:


# Get the URI of the output artifact representing the transformed examples, which is a directory
train_uri = os.path.join(transform.outputs['transformed_examples'].get()[0].uri, 'Split-train')

# Get the list of files in this directory (all compressed TFRecord files)
tfrecord_filenames = [os.path.join(train_uri, name)
                      for name in os.listdir(train_uri)]

# Create a `TFRecordDataset` to read these files
dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")


# In[18]:


# Define a helper function to get individual examples
def get_records(dataset, num_records):
    '''Extracts records from the given dataset.
    Args:
        dataset (TFRecordDataset): dataset saved by ExampleGen
        num_records (int): number of records to preview
    '''
    
    # initialize an empty list
    records = []
    
    # Use the `take()` method to specify how many records to get
    for tfrecord in dataset.take(num_records):
        
        # Get the numpy property of the tensor
        serialized_example = tfrecord.numpy()
        
        # Initialize a `tf.train.Example()` to read the serialized data
        example = tf.train.Example()
        
        # Read the example data (output is a protocol buffer message)
        example.ParseFromString(serialized_example)
        
        # convert the protocol bufffer message to a Python dictionary
        example_dict = (MessageToDict(example))
        
        # append to the records list
        records.append(example_dict)
        
    return records


# You should see from the output of the cell below that the transformed raw image (i.e. `image_raw_xf`) now has a float array that is scaled from 0 to 1. Similarly, you'll see that the transformed label (i.e. `label_xf`) is now one-hot encoded.

# In[19]:


# Get 1 record from the dataset
sample_records = get_records(dataset, 1)

# Print the output
pp.pprint(sample_records)


# ### Wrap Up
# 
# This notebook demonstrates how to do feature engineering with image datasets as opposed to simple tabular data. This should come in handy in your computer vision projects and you can also try replicating this process with other image datasets from [TFDS](https://www.tensorflow.org/datasets/catalog/overview#image_classification).
